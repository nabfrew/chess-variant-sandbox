"""R-L4b stage 3 (`docs/reference/capture-effects-design.md` §4, §6): ``rules.capture``, the spelling.

What a capture does is one ordered list (`spec/capture.py`). The old keys say its default order;
`capture.capture_list` reads either spelling, the normal form and the converter write a list the
old keys can say back as them (so no stored game moves its name or its tables), and a list only
the new key says is its own game, compiled from each kind's rule with the list handed to the
core. The 18 pairs §3.1 and §3.2 lift are accepted, and each plays as the list.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec, with_capture_list
from chess_variant_sandbox.fuzz_lifted import generate, keys
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.capture import (
    as_list_spelling,
    capture_as_old_keys,
    capture_base,
    capture_list,
    compile_entries,
    lifted_pairs,
    needs_the_walk,
)
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, validate_spec
from chess_variant_sandbox.spec.spelling import current_spelling

ROOT = Path(__file__).resolve().parent.parent
STORED = sorted(
    str(path.relative_to(ROOT)).replace("\\", "/")
    for folder in ("rulesets", "archive", "unlike") for path in (ROOT / folder).glob("*.json")
)


def _load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _six(**rules: Any) -> dict[str, Any]:
    spec = copy.deepcopy(generate.SIX)
    spec["rules"].update(rules)
    return spec


def _demoting(position: str = "stay") -> dict[str, Any]:
    return {"effect": "demote", "position": position, "immune": ["K"],
            "chain": [{"from": "R", "to": "B"}, {"from": "B", "to": "N"}, {"from": "N", "to": "P"}]}


class TestEveryStoredGameKeepsItsNameAndTables:
    """§4's proof: a re-spelling. Every file under `rulesets/`, `archive/` and `unlike/` written
    as ``rules.capture`` is the same game: one identity, one set of tables, one verdict, and the
    converter writes it back to the file's own spelling. (`test_schema_nine.FROZEN` holds every
    file's identity and tables unchanged as written.)"""

    @pytest.mark.parametrize("path", STORED)
    def test_the_list_spelling_is_the_same_game(self, path: str) -> None:
        spec = _load(path)
        listed = as_list_spelling(spec)
        if listed is None:
            assert capture_list(spec) == [], path  # no capture rule: plain capture
            return
        assert capture_as_old_keys(listed) is not None, path
        assert spec_hash(listed) == spec_hash(spec)
        assert validate_spec(listed) == validate_spec(spec)
        assert current_spelling(listed) == current_spelling(spec)
        assert not needs_the_walk(spec) and not needs_the_walk(listed)
        if not validate_spec(spec):
            assert tables_fingerprint(listed) == tables_fingerprint(spec)

    def test_the_spelling_moved_no_version(self) -> None:
        """No older file needs the rewrite, so the stamp stays (CLAUDE.md rule 1)."""
        assert SCHEMA_VERSION == 31

    def test_some_stored_games_say_a_capture_rule(self) -> None:
        said = [path for path in STORED if as_list_spelling(_load(path)) is not None]
        assert len(said) >= 30


class TestTheListSpelling:
    def test_a_list_the_old_keys_say_is_written_back_as_them(self) -> None:
        listed = _six(capture=[{"do": "blast", "radius": 1, "immune": ["K"]}])
        old = _six(explode_on_capture={"radius": 1, "immune": ["K"]})
        assert validate_spec(listed) == []
        assert capture_as_old_keys(listed) == old
        assert spec_hash(listed) == spec_hash(old)
        assert tables_fingerprint(listed) == tables_fingerprint(old)
        assert current_spelling(listed)["rules"].get("explode_on_capture") == {"radius": 1, "immune": ["K"]}
        assert "capture" not in current_spelling(listed)["rules"]

    def test_the_order_is_the_meaning(self) -> None:
        hand = {"drops": {}}
        first = _six(capture=[{"do": "hand"}, {"do": "strike", **_demoting()}], **hand)
        second = _six(capture=[{"do": "strike", **_demoting()}, {"do": "hand"}], **hand)
        assert validate_spec(first) == [] and validate_spec(second) == []
        assert capture_as_old_keys(first) is not None  # the default order: the old keys say it
        assert capture_as_old_keys(second) is None  # a strike ahead of the hand: only the list
        assert spec_hash(first) != spec_hash(second)
        assert "capture" in current_spelling(second)["rules"]
        assert needs_the_walk(first) and needs_the_walk(second)  # T2: the core walks both

    def test_a_list_only_the_key_says_compiles_its_rules_and_hands_the_list_over(self) -> None:
        spec = _six(capture=[{"do": "strike", **_demoting()}, {"do": "hand"}], drops={})
        compiled = compile_spec(spec)
        assert [e.do for e in compiled.capture or ()] == ["strike", "hand"]
        by_hand = with_capture_list(compile_spec(capture_base(spec), compile_entries(capture_list(spec) or [])),
                                    compile_entries(capture_list(spec) or []))
        assert compiled.capture == by_hand.capture
        engine = Engine(spec)
        state = engine.initial_state()
        for _ in range(10):
            moves = engine.legal_moves(state)
            if not moves or engine.terminal(state) is not None:
                break
            state = engine.apply(state, moves[-1])

    def test_on_path_by_spread_and_spares_taken(self) -> None:
        spec = _six(capture=[
            {"do": "blast", "radius": 1, "immune": ["K"], "spares_taken": True},
            {"do": "strike", **_demoting(), "on": ["landing", "blast", "collision"]},
        ])
        assert validate_spec(spec) == []
        entries = capture_list(spec)
        assert entries is not None and entries[0]["spares_taken"] and entries[1]["on"] == ["landing", "blast", "collision"]
        # The two switches say it: the old keys' `strikes: "both"`.
        lowered = capture_as_old_keys(spec)
        assert lowered is not None and lowered["rules"]["explode_on_capture"]["strikes"] == "both"

    @pytest.mark.parametrize(("rules", "words"), [
        ({"capture": [{"do": "blast", "radius": 1}], "explode_on_capture": {"radius": 1}},
         "two spellings"),
        ({"capture": [{"do": "hand"}], "drops": {"from_captures": True}}, "from_captures beside"),
        ({"capture": [{"do": "hand"}]}, "give the hand"),
        ({"capture": [{"do": "pay"}]}, "no bounty declared"),
        ({"capture": [{"do": "nothing"}]}, "do must be one of"),
        ({"capture": [{"do": "strike", **_demoting(), "on": []}]}, "reads no taking"),
        ({"capture": [{"do": "strike", **_demoting(), "path_by": ["R"]}]}, "never sweeps"),
        ({"capture": [{"do": "strike", **_demoting()}, {"do": "strike", "effect": "promote"}]},
         "states its kind's rule again"),
        ({"capture": [{"do": "blast", "radius": 1, "spread": True, "on": ["landing"]}]}, "spread centres"),
        ({"capture": [{"do": "strike", **_demoting()}, {"do": "blast", "radius": 1, "immune": ["K"]}],
          "simultaneous": {}, "win": "royal_capture"}, "written before the list"),
    ])
    def test_what_the_list_refuses(self, rules: dict[str, Any], words: str) -> None:
        errors = validate_spec(_six(**rules))
        assert any(words in error for error in errors), errors

    def test_a_bounty_needs_a_pay_entry(self) -> None:
        spec = _six(capture=[{"do": "blast", "radius": 1, "immune": ["K"]}], scoring={"count": "points"})
        spec["pieces"][1]["bounty"] = 3
        assert any("never paid" in error for error in validate_spec(spec))
        spec["rules"]["capture"].append({"do": "pay"})
        assert validate_spec(spec) == []


class TestTheEighteenPairsAreAccepted:
    """§3.1 and §3.2: none of the 18 is refused, the old keys' spelling of each hands the core
    its default-order list (`capture.lifted_pairs`), and a case the fuzz builds for it plays."""

    # R-L4b5's two round rows (T1.rounds, T4.rounds) are `test_capture_list_in_rounds.py`'s.
    ROWS = [row for row in keys.lifted() if row.by == "R-L4b" and not row.id.endswith(".rounds")]

    def test_eighteen_rows(self) -> None:
        assert len(self.ROWS) == 18

    @pytest.mark.parametrize("row", ROWS, ids=lambda row: row.id)
    def test_the_fuzz_builds_and_plays_it(self, row: keys.Lifted) -> None:
        for seed in range(80):
            case = generate.case(seed, (row,))
            if case.spec is not None:
                break
        else:
            pytest.fail(f"no case built for {row.id}")
        assert validate_spec(case.spec) == []
        assert row.id in keys.pairs_present(case.spec)
        engine = Engine(case.spec)
        state = engine.initial_state()
        for _ in range(8):
            if engine.terminal(state) is not None:
                break
            if engine.is_round(state):  # R-L4b5: a pair may now play in rounds
                pair = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
                if not pair[0] or not pair[1]:
                    break
                state = engine.apply_round(state, pair[0][0], pair[1][0])
                continue
            moves = engine.legal_moves(state)
            if not moves:
                break
            state = engine.apply(state, moves[0])

    @pytest.mark.parametrize(("rules", "pair"), [
        ({"strike": _demoting("swap"), "explode_on_capture": {"radius": 1, "immune": ["K"], "strikes": "target"}},
         "T1.position"),
        ({"strike": _demoting(), "drops": {"from_captures": True}}, "T2"),
        ({"strike": _demoting(), "contacts": {"links": [{"mover": "N", "onto": "B", "mover_to": "B"}],
                                              "immune": ["K"]}}, "T3"),
        ({"contacts": {"links": [{"mover": "N", "onto": "B", "mover_to": "B"}], "immune": ["K"]},
          "explode_on_capture": {"radius": 1, "immune": ["K"], "strikes": "blast"}}, "T4.switch"),
        ({"contacts": {"links": [{"mover": "N", "onto": "B", "onto_to": "P"}], "position": "stay",
                       "immune": ["K"]}, "drops": {"from_captures": True}}, "T5"),
    ])
    def test_the_old_keys_of_a_lifted_pair_hand_over_their_list(self, rules: dict[str, Any], pair: str) -> None:
        spec = _six(**rules)
        assert validate_spec(spec) == [], validate_spec(spec)
        assert pair in lifted_pairs(spec)
        compiled = compile_spec(spec)
        assert compiled.capture is not None
        assert [e.do for e in compiled.capture] == [e["do"] for e in capture_list(spec) or []]

    def test_a_contact_table_placed_choose_beside_a_strike_stays_refused(self) -> None:
        spec = _six(strike=_demoting(), contacts={
            "links": [{"mover": "N", "onto": "B", "onto_to": "P"}], "position": "choose",
            "landing": {"vectors": [[0, 1]], "symmetry": "rotate4"}, "immune": ["K"]})
        assert "T3" in refused_ids(validate_spec(spec))

    def test_a_lifted_pair_with_a_shot_in_a_round_stays_refused(self) -> None:
        """R-L4b5: a round walks the list, but a shot is written before it."""
        spec = _six(strike=_demoting(), drops={"from_captures": True}, simultaneous={}, win="royal_capture")
        assert any("written before the list" in error for error in validate_spec(spec))

    def test_a_lifted_pair_in_a_round_plays(self) -> None:
        """R-L4b5: the same pair with the struck piece sent home is a list a round walks."""
        spec = _six(strike=_demoting("home"), drops={"from_captures": True}, simultaneous={}, win="royal_capture")
        assert validate_spec(spec) == [], validate_spec(spec)
