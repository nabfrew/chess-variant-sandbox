"""Schema 11: a contact link onto the mover's own piece (Q19b).

`rules.contacts.links[].own` lets a piece move onto a friendly piece along its capture
moves, and the link says what both become. It is the one M2 feature that adds a third arm
to the generator, so it moves perft and takes a version of its own.

The inertness proof is `test_schema_nine.py::TestTheBumpMovedNothing`, which pins the
mechanics hash and tables fingerprint of every committed and archived ruleset. This file
holds the feature itself: gated, refused where its meaning is unsettled, played by the
core, and read by every surface `new-move-primitive` names.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables, rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.spec.describe import _describe_contacts
from chess_variant_sandbox.spec.distance import _SCALES, distance, features
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.schema import SCHEMA_VERSION, _schema_eleven_uses

ROOT = Path(__file__).resolve().parents[1]

MERGE = {"mover": "P", "onto": "R", "own": True, "mover_to": "Q"}


@pytest.fixture(scope="module")
def chess() -> dict[str, Any]:
    return json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def merger() -> dict[str, Any]:
    return json.loads((ROOT / "archive" / "merger.json").read_text(encoding="utf-8"))


def _contacts(
    chess: dict[str, Any], links: list[dict[str, Any]], position: str = "advance"
) -> dict[str, Any]:
    """chess plus a contact table, stamped for schema 11."""
    child = copy.deepcopy(chess)
    child["schema_version"] = 11
    child["rules"]["contacts"] = {"position": position, "immune": ["K"], "links": links}
    return child


def _square(name: str, width: int = 8) -> int:
    return (int(name[1:]) - 1) * width + ord(name[0]) - ord("a")


def _moves_between(engine: Engine, state: Any, origin: str, target: str) -> list[int]:
    wanted = (_square(origin), _square(target))
    return [m for m in engine.legal_moves(state) if (move_from(m), move_to(m)) == wanted]


def _play(engine: Engine, line: list[tuple[str, str]]) -> Any:
    state = engine.initial_state()
    for origin, target in line:
        (move,) = _moves_between(engine, state, origin, target)
        state = engine.apply(state, move)
    return state


# White opens the a-file and lifts the rook to a3, where the b-pawn's capture reaches it.
_LIFT = [("a2", "a4"), ("h7", "h6"), ("a1", "a3"), ("h6", "h5")]


def _occupant(engine: Engine, state: Any, square: str) -> tuple[int, str] | None:
    value = state.board[_square(square)]
    if value == 0:
        return None
    return engine.spec.owner_of(value), engine.spec.piece_ids[engine.spec.type_of(value)]


class TestTheVersionIsBumped:
    def test_the_schema_version_is_at_least_eleven(self) -> None:
        assert SCHEMA_VERSION >= 11

    def test_the_collector_names_the_one_feature(
        self, chess: dict[str, Any], merger: dict[str, Any]
    ) -> None:
        assert _schema_eleven_uses(chess) == set()
        enemy = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "Q"}])
        assert _schema_eleven_uses(enemy) == set()
        assert _schema_eleven_uses(merger) == {"rules.contacts.links[].own"}

    def test_an_older_stamp_is_refused_by_name(self, merger: dict[str, Any]) -> None:
        spec = copy.deepcopy(merger)
        spec["schema_version"] = 10
        errors = validate_spec(spec)
        assert any("links[].own" in e and "schema 11" in e for e in errors), errors


class TestTheValidator:
    def test_the_showcase_validates(self, merger: dict[str, Any]) -> None:
        assert validate_spec(merger) == []

    def test_false_is_a_second_spelling_of_absent(self, chess: dict[str, Any]) -> None:
        errors = validate_spec(_contacts(chess, [dict(MERGE, own=False)]))
        assert any("own must be true" in e for e in errors), errors

    def test_an_own_link_onto_an_immune_piece_is_refused(self, chess: dict[str, Any]) -> None:
        """Every royal must be immune, so this is also what keeps a king off the target."""
        errors = validate_spec(_contacts(chess, [dict(MERGE, onto="K")]))
        assert any("which rules.contacts.immune says no contact reaches" in e for e in errors)

    def test_it_is_refused_beside_must_capture(self, chess: dict[str, Any]) -> None:
        spec = _contacts(chess, [MERGE])
        spec["rules"]["must_capture"] = True
        errors = validate_spec(spec)
        assert any("rules.must_capture" in e and "own piece" in e for e in errors), errors

    def test_an_own_link_has_to_change_something(self, chess: dict[str, Any]) -> None:
        errors = validate_spec(_contacts(chess, [{"mover": "P", "onto": "R", "own": True}]))
        assert any("a link onto an own piece has to change one" in e for e in errors), errors

    def test_an_own_and_an_enemy_link_for_one_pair_are_two_meetings(
        self, chess: dict[str, Any]
    ) -> None:
        both = [MERGE, {"mover": "P", "onto": "R", "mover_to": "N"}]
        assert validate_spec(_contacts(chess, both)) == []
        errors = validate_spec(_contacts(chess, [MERGE, dict(MERGE, mover_to="B")]))
        assert any("onto its own 'R'" in e and "one meeting" in e for e in errors), errors


class TestTheCompiledTables:
    def test_a_table_without_own_sends_no_own_bytes(self, chess: dict[str, Any]) -> None:
        enemy = compile_spec(_contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "N"}]))
        assert enemy.strike is not None and not enemy.strike.meets_own
        tables = build_tables(enemy)
        assert tables["contact_own_onto_to"] == [] and tables["contact_own_mover_to"] == []
        assert not {"contact_own_onto_to", "contact_own_mover_to"} & set(
            canonical_tables(compile_spec(chess))
        )

    def test_an_own_link_fills_its_own_cell_and_no_other(self, chess: dict[str, Any]) -> None:
        compiled = compile_spec(_contacts(chess, [MERGE]))
        strike = compiled.strike
        assert strike is not None and strike.meets_own and strike.compact_to is None
        ids = list(compiled.piece_ids)
        n = compiled.n_types
        cell = ids.index("P") * n + ids.index("R")
        assert strike.own_mover_to[cell] == ids.index("Q")
        assert [i for i, v in enumerate(strike.own_mover_to) if v >= 0] == [cell]
        assert all(v < 0 for v in strike.mover_to + strike.onto_to + strike.own_onto_to)
        tables = build_tables(compiled)
        assert len(tables["contact_own_mover_to"]) == n * n
        assert rust_supports(compiled) is None

    def test_an_own_link_that_crowns_is_a_second_royal(self, chess: dict[str, Any]) -> None:
        crowned = compile_spec(_contacts(chess, [dict(MERGE, mover_to="K")]))
        assert crowned.multi_royal
        assert compile_spec(_contacts(chess, [MERGE])).multi_royal is False


class TestTheCorePlaysIt:
    def test_the_merge_is_offered_only_where_the_link_says(
        self, chess: dict[str, Any], merger: dict[str, Any]
    ) -> None:
        plain, merging = Engine(chess), Engine(merger)
        before_plain, before_merge = _play(plain, _LIFT), _play(merging, _LIFT)
        assert _moves_between(plain, before_plain, "b2", "a3") == []
        assert len(_moves_between(merging, before_merge, "b2", "a3")) == 1
        # Nothing else changes: the one extra move is the merge.
        assert len(merging.legal_moves(before_merge)) == len(plain.legal_moves(before_plain)) + 1
        # A knight has no own link, so the rook still blocks its landing square.
        assert _moves_between(merging, before_merge, "b1", "a3") == []

    def test_the_merge_leaves_a_queen_and_no_rook(self, merger: dict[str, Any]) -> None:
        engine = Engine(merger)
        state = _play(engine, [*_LIFT, ("b2", "a3")])
        assert _occupant(engine, state, "a3") == (0, "Q")
        assert _occupant(engine, state, "b2") is None
        assert state.key == engine._compute_hash(state)
        (move,) = _moves_between(engine, _play(engine, _LIFT), "b2", "a3")
        assert engine.describe_move(_play(engine, _LIFT), move) == "Pb2xa3/P->Q"

    def test_perft_matches_chess_until_a_merge_is_possible(
        self, chess: dict[str, Any], merger: dict[str, Any]
    ) -> None:
        """No white pawn can reach its own rook in four plies, so the counts agree there."""
        plain, merging = Engine(chess), Engine(merger)
        for depth in (1, 2, 3, 4):
            assert merging.perft(merging.initial_state(), depth) == plain.perft(
                plain.initial_state(), depth
            )

    def test_the_legality_filter_refuses_a_merge_off_a_pin(
        self, merger: dict[str, Any]
    ) -> None:
        """A pinned pawn may not merge onto its own rook off the pin line."""

        def position(with_bishop: bool) -> dict[str, Any]:
            spec = copy.deepcopy(merger)
            spec["rules"].pop("castling")
            placements = [
                {"piece": "K", "squares": ["e1"]},
                {"piece": "P", "squares": ["d2"]},
                {"piece": "R", "squares": ["e3"]},
                {"piece": "K", "player": 1, "squares": ["h8"]},
            ]
            if with_bishop:
                placements.append({"piece": "B", "player": 1, "squares": ["a5"]})
            spec["setup"] = {"mirror": False, "placements": placements}
            assert validate_spec(spec) == [], validate_spec(spec)
            return spec

        free = Engine(position(False))
        assert len(_moves_between(free, free.initial_state(), "d2", "e3")) == 1
        pinned = Engine(position(True))
        assert _moves_between(pinned, pinned.initial_state(), "d2", "e3") == []

    @pytest.mark.parametrize(
        "position, link",
        [
            ("advance", MERGE),
            ("stay", {"mover": "N", "onto": "P", "own": True, "onto_to": "B"}),
            ("swap", {"mover": "B", "onto": "P", "own": True, "mover_to": "N", "onto_to": "B"}),
        ],
    )
    def test_a_short_game_keeps_its_incremental_hash(
        self, chess: dict[str, Any], position: str, link: dict[str, Any]
    ) -> None:
        spec = _contacts(chess, [link], position=position)
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        rng = random.Random(19)
        state = engine.initial_state()
        for _ in range(40):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            for move in moves:
                engine.apply(state, move)
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), "hash drifted"


class TestTheOtherSurfaces:
    def test_the_description_says_the_mover_may_move_onto_its_own_piece(
        self, merger: dict[str, Any]
    ) -> None:
        lines = _describe_contacts(merger, merger["rules"])
        said = "pawn moves onto its own rook, the pawn becomes queen"
        assert any(said in line for line in lines), lines
        assert any("one of its own pieces" in line for line in lines), lines

    def test_it_has_a_distance_feature_of_its_own(self, chess: dict[str, Any]) -> None:
        assert "contacts_meet_own" in _SCALES
        own = _contacts(chess, [MERGE])
        enemy = _contacts(chess, [{"mover": "P", "onto": "R", "mover_to": "Q"}])
        assert features(own)["contacts_meet_own"] == 1.0
        assert features(enemy)["contacts_meet_own"] == 0.0
        assert distance(own, enemy) > 0.0

    def test_fairy_refuses_it_by_name(self, merger: dict[str, Any]) -> None:
        with pytest.raises(Untranslatable) as refusal:
            translate(compile_spec(merger))
        assert any("rules.contacts" in reason for reason in refusal.value.reasons)

    def test_the_json_schema_offers_it_to_a_proposer(self) -> None:
        contacts = rule_spec_schema()["properties"]["rules"]["properties"]["contacts"]
        link = contacts["properties"]["links"]["items"]
        assert link["properties"]["own"]["type"] == "boolean"


# ------------------------------------------------ Q922: the contact table's pair axis


def _contact_axes(links: list[dict[str, Any]]) -> dict[str, float]:
    from chess_variant_sandbox.spec.distance import _contact_features

    pieces = [{"id": i, "value": v} for i, v in (("P", 1), ("N", 3), ("B", 3))]
    return _contact_features(pieces, {"position": "stay", "links": links})


class TestTheVampireAxisKeysOnWhoMeetsWhat:
    """One victim left as two different things is `contacts_by_pair`, unless the two links
    are on different sides of the `own` line: a contact with your own piece is another rule.
    """

    @pytest.mark.parametrize(
        ("second", "by_pair"),
        [
            ({"mover": "B", "onto": "N", "onto_to": "B"}, 1.0),            # same victim, another result
            ({"mover": "B", "onto": "N", "onto_to": "P", "own": True}, 0.0),  # own is a rule apart
            ({"mover": "B", "onto": "N", "onto_to": "B", "own": False}, 1.0),  # False is no own
            ({"mover": "B", "onto": "B", "onto_to": "B"}, 0.0),            # another victim
        ],
        ids=["same", "own", "explicit-false", "other-victim"],
    )
    def test_the_pair_axis(self, second: dict[str, Any], by_pair: float) -> None:
        first = {"mover": "P", "onto": "N", "onto_to": "P"}
        assert _contact_axes([first, second])["contacts_by_pair"] == by_pair

    def test_two_own_links_to_one_victim_are_a_pair(self) -> None:
        links = [
            {"mover": "P", "onto": "N", "onto_to": "P", "own": True},
            {"mover": "B", "onto": "N", "onto_to": "B", "own": True},
        ]
        assert _contact_axes(links)["contacts_by_pair"] == 1.0


class TestAPerPlayerSeamIsReadOnlyWhenItSaysSomething:
    """`wrap_stated` falls back to the shared word for a player the table leaves out, and
    to the shared word for both when the table is empty or not a table at all."""

    @pytest.mark.parametrize(
        ("wrap", "files"),
        [
            ({"files": "wrap"}, ("wrap", "wrap")),
            ({"files": "wrap", "files_by_player": {}}, ("wrap", "wrap")),
            ({"files": "wrap", "files_by_player": ["x"]}, ("wrap", "wrap")),
            ({"files": "wrap", "files_by_player": "x"}, ("wrap", "wrap")),
            ({"files": "wrap", "files_by_player": {"1": "none"}}, ("wrap", "none")),
            ({"files": "wrap", "files_by_player": {"0": "none"}}, ("none", "wrap")),
            ({"files_by_player": {"0": "wrap", "1": "none"}}, ("wrap", "none")),
            ({}, ("none", "none")),
            ("junk", ("none", "none")),
        ],
        ids=str,
    )
    def test_the_two_players_words(self, wrap: Any, files: tuple[str, str]) -> None:
        from chess_variant_sandbox.spec.schema import wrap_stated

        assert wrap_stated(wrap)["files"] == files
