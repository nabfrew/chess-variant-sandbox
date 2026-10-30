"""The compiler's second-royal read: every route to two royals sets ``multi_royal``.

``CompiledSpec.multi_royal`` is what takes the core off the single-royal fast path --
``royal_square`` answering "where is it?" with one ``find`` -- and onto scanning every
royal. False for a game that can put two kings on one side is a silently wrong game: under
``royal_capture`` the ending still scans, but ``in_check`` and the check-aware ordering see
one king; under ``checkmate`` the legality filter guards one king and leaves the other en
prise. No perft can see it, because nothing changes a move count until the second king
exists.

The read used to consult the spec a second time and knew the dict form of ``promotion``
only, so a promotion *list* ending on a king and a strike chain ending on a king both
compiled ``False`` (``docs/game-ideas.md``, costing table row 2 and its note 2). It reads
the compiled rules now -- one interpretation, the one the core is handed. The specs below
are that row's spellings, recreated so that ``spelling_hash`` gives the hashes the note names,
plus the route the compiled read finds that the note did not: a hand a royal can reach.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec, compile_spec
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.canonical import spelling_hash
from chess_variant_sandbox.spec.schema import royalty_can_multiply

ROOT = Path(__file__).resolve().parents[1]

QUEEN_TO_KING = [
    {"piece": "P", "rank": "last", "to": ["Q", "R", "B", "N"]},
    {"piece": "Q", "rank": "last", "to": ["K"]},
]


def chess(version: int = 1, **rules: Any) -> dict[str, Any]:
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    spec["schema_version"] = version
    # Q1069a: chess as the costing table named it, before it drew on dead material.
    spec["rules"].pop("insufficient_material", None)
    spec["rules"].update(rules)
    return spec


def queen_promotes_by_rank(win: str) -> dict[str, Any]:
    """Row 2's rank spelling. ``en_passant`` goes: it is refused beside a promotion list."""
    spec = chess(2, win=win, promotion=QUEEN_TO_KING)
    del spec["rules"]["en_passant"]
    return spec


def queen_promotes_by_strike() -> dict[str, Any]:
    """Row 2's strike spelling: the archived ``hierarchy-rise`` chain, one link longer."""
    spec = load_spec(ROOT / "archive" / "hierarchy-rise.json")
    # The file's stamp when the note costed it; Q1008b re-spelt the file at the current one.
    spec["schema_version"] = 6
    spec["rules"]["win"] = "royal_capture"
    spec["rules"]["strike"]["chain"].append({"from": "Q", "to": "K"})
    return spec


def compiled(spec: dict[str, Any]) -> CompiledSpec:
    assert validate_spec(spec) == [], validate_spec(spec)
    return compile_spec(spec)


class TestRowTwoSpellings:
    """The three specs ``game-ideas.md`` costed, by hash, and the flag each now carries."""

    @pytest.mark.parametrize(
        ("build", "digest"),
        [
            (lambda: queen_promotes_by_rank("royal_capture"), "fa599a1bc95c"),
            (lambda: queen_promotes_by_rank("checkmate"), "43552cf7d30c"),
            (queen_promotes_by_strike, "2320c64d75c1"),
        ],
        ids=["promotion-list", "promotion-list-under-checkmate", "strike-chain"],
    )
    def test_the_spelling_is_the_one_costed_and_reads_two_royals(
        self, build: Callable[[], dict[str, Any]], digest: str
    ) -> None:
        spec = build()
        # The note named each spelling by the hash of the time, the spelling hash; since
        # Q1008a a game's name is its normal form's, and these were never stored games.
        assert spelling_hash(spec) == digest, "not the spec the costing table names"
        assert compiled(spec).multi_royal

    def test_under_checkmate_the_second_king_is_guarded(self) -> None:
        """The wrong game the note describes, not played.

        Two white kings, the *later* one in scan order attacked: a single-royal read finds
        a1, calls the side safe, and lets the a1 king wander while e1 stands en prise.
        """
        engine = Engine(queen_promotes_by_rank("checkmate"))
        state = engine.make_state(
            {"a1": ("K", 0), "e1": ("K", 0), "e8": ("R", 1), "h8": ("K", 1)}
        )
        assert engine.in_check(state), "a rook on the e-file attacks the e1 king"
        assert "Ka1-b1" not in engine.move_texts(state), "and the other king may not ignore it"


class TestEveryRoute:
    """Each way a royal enters play after ply zero, and the spellings that only look like one."""

    def test_a_dict_promotion_still_counts(self) -> None:
        spec = chess(2, promotion={"piece": "P", "rank": "last", "to": ["K"]})
        assert compiled(spec).multi_royal

    def test_targets_stated_per_player(self) -> None:
        spec = chess(
            2,
            promotion={
                "piece": "P", "rank": "last", "to_by_player": {"0": ["Q"], "1": ["K"]}
            },
        )
        assert compiled(spec).multi_royal

    def test_a_captured_piece_that_returns_to_the_hand_as_a_royal(self) -> None:
        """``demotions`` beside ``drops``: a queen ``promoted_from`` the king is dropped back
        as a king, so the captor of a queen holds two."""
        spec = chess(2, drops={"from_captures": True})
        queen = next(p for p in spec["pieces"] if p["id"] == "Q")
        queen["promoted_from"] = "K"
        assert compiled(spec).multi_royal

    def test_a_capturable_royal_beside_drops(self) -> None:
        """Where taking a king leaves the game running, the king itself reaches the hand."""
        spec = chess(2, win="none", bare_royal="loss", drops={"from_captures": True})
        assert compiled(spec).multi_royal

    @pytest.mark.parametrize(
        "path",
        ["rulesets/chess.json", "rulesets/crazyhouse.json", "rulesets/dobutsu.json",
         "archive/hierarchy-rise.json"],
        ids=lambda p: Path(p).stem,
    )
    def test_no_route_in_a_game_with_one_king(self, path: str) -> None:
        """Drops under ``checkmate`` never hand over a king; a struck queen stays a queen."""
        assert not compiled(load_spec(ROOT / path)).multi_royal

    def test_a_link_to_the_king_closed_by_immunity(self) -> None:
        """An immune source is a plain capture, so its link never fires."""
        spec = queen_promotes_by_strike()
        spec["rules"]["strike"]["immune"].append("Q")
        assert not compiled(spec).multi_royal

    def test_a_chain_under_effect_capture_is_no_chain(self) -> None:
        """Refused since Q538 as a chain the core never plays, and still no route to
        the validator's reading, which meets it before that refusal decides anything."""
        from chess_variant_sandbox.spec.schema import PLAYED_AS_CAPTURE

        spec = strike_under_effect_capture()
        assert any(PLAYED_AS_CAPTURE in error for error in validate_spec(spec))
        assert not royalty_can_multiply(spec)


def strike_closed_by_immunity() -> dict[str, Any]:
    spec = queen_promotes_by_strike()
    spec["rules"]["strike"]["immune"].append("Q")
    return spec


def strike_under_effect_capture() -> dict[str, Any]:
    spec = queen_promotes_by_strike()
    spec["rules"]["strike"]["effect"] = "capture"
    return spec


def queen_returns_from_the_hand_as_a_king() -> dict[str, Any]:
    spec = chess(2, drops={"from_captures": True})
    next(p for p in spec["pieces"] if p["id"] == "Q")["promoted_from"] = "K"
    return spec


# One spec per route, in the order `royalty_can_multiply` reads them, plus the spelling
# that only looks like a route and one game with no route at all. The second such
# spelling, a chain under `effect: capture`, is refused since Q538 and is held by
# `TestEveryRoute.test_a_chain_under_effect_capture_is_no_chain` instead.
ROUTE_SPECS: list[tuple[str, Callable[[], dict[str, Any]]]] = [
    ("no-route", lambda: chess(2)),
    ("promotion-list", lambda: queen_promotes_by_rank("royal_capture")),
    ("promotion-list-under-checkmate", lambda: queen_promotes_by_rank("checkmate")),
    ("promotion-dict", lambda: chess(2, promotion={"piece": "P", "rank": "last", "to": ["K"]})),
    (
        "promotion-per-player",
        lambda: chess(
            2,
            promotion={"piece": "P", "rank": "last", "to_by_player": {"0": ["Q"], "1": ["K"]}},
        ),
    ),
    ("strike-chain", queen_promotes_by_strike),
    ("strike-chain-immune", strike_closed_by_immunity),
    ("hand-holds-a-demoted-royal", queen_returns_from_the_hand_as_a_king),
    (
        "hand-holds-the-royal-itself",
        lambda: chess(2, win="none", bare_royal="loss", drops={"from_captures": True}),
    ),
]


class TestTheValidatorReadsTheSameRoutes:
    """One rule, two vocabularies, held together by these tests.

    ``spec.schema.royalty_can_multiply`` answers the same question as
    ``engine.compile._royalty_can_multiply`` off the spec rather than off the compiled
    tables, because the schema-2 gate has to refuse a schema-1 spec *before* anything is
    compiled and the compiler imports the validator, not the other way round. Two
    readings of one rule is what CLAUDE.md's fourth rule exists to prevent, so they are
    pinned against each other here the way ``fairy/verify.py`` pins the only other pair:
    every route and both spellings that only look like one. The whole committed tree is
    held by ``test_schema_two.py::test_every_committed_ruleset_plays_a_game_through``,
    which compiles every committed spec anyway (Q377).

    A disagreement is not cosmetic in either direction. The validator reading *less*
    than the compiler lets a schema-1 spec through that the core will play as a
    two-royal game; reading *more* refuses a game schema 1 was always entitled to.
    """

    @pytest.mark.parametrize(
        ("build",), [(build,) for _, build in ROUTE_SPECS], ids=[name for name, _ in ROUTE_SPECS]
    )
    def test_the_two_readings_agree(self, build: Callable[[], dict[str, Any]]) -> None:
        spec = build()
        assert royalty_can_multiply(spec) is compiled(spec).multi_royal


# ------------------------------------------------ Q922: every branch of the spelling-level read


def _bare(pieces: list[Any], setup: Any = None, **rules: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"pieces": pieces, "rules": rules}
    if setup is not None:
        spec["setup"] = setup
    return spec


KING = {"id": "K", "royal": True}
QUEEN = {"id": "Q"}
RAVEN = {"id": "R", "royal": True}


def _placed(*placements: Any) -> dict[str, Any]:
    return {"placements": list(placements)}


# Each case is a spec the spelling-level read must answer, with the answer.
SPELLING_CASES = {
    "no royal at all": (_bare([QUEEN]), False),
    "one royal, nothing else": (_bare([KING, QUEEN]), False),
    "two royal types": (_bare([KING, RAVEN]), True),
    "pieces that are not a list": ({"pieces": "KQ", "rules": {}}, False),
    "rules that are not a dict": ({"pieces": [KING], "rules": []}, False),
    "one king placed on one square": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1"]})), False),
    "one king on two squares": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1", "e2"]})), True),
    "one king on three squares": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1", "e2", "e3"]})), True),
    "two placements of one king, one square each, one player": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1"]},
                              {"piece": "K", "squares": ["e2"]})), True),
    "a placement that names the first player and one that leaves it out": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1"], "player": 0},
                              {"piece": "K", "squares": ["e2"]})), True),
    "one king for each player": (
        _bare([KING], _placed({"piece": "K", "squares": ["e1"], "player": 0},
                              {"piece": "K", "squares": ["e8"], "player": 1})), False),
    "a non-royal placement ahead of two kings": (
        _bare([KING, QUEEN], _placed({"piece": "Q", "squares": ["d1"]},
                                     "junk",
                                     {"piece": "K", "squares": ["e1", "e2"]})), True),
    "a king whose squares are not a list": (
        _bare([KING], _placed({"piece": "K", "squares": "e1"})), False),
    "a setup that is not a dict": (_bare([KING], "e1"), False),
    "placements that are not a list": (_bare([KING], {"placements": "e1"}), False),
    "a contact that makes the mover a royal": (
        _bare([KING, QUEEN], contacts={"links": ["junk", {"mover": "Q", "onto": "K", "mover_to": "K"}]}),
        True),
    "a contact that makes the piece met a royal": (
        _bare([KING, QUEEN], contacts={"links": [{"mover": "Q", "onto": "Q", "onto_to": "K"}]}),
        True),
    "a contact that makes an immune piece a royal": (
        _bare([KING, QUEEN],
              contacts={"links": [{"mover": "Q", "onto": "Q", "onto_to": "K"}], "immune": ["Q"]}),
        False),
    "a contact that makes the piece met a royal, another type immune": (
        _bare([KING, QUEEN],
              contacts={"links": [{"mover": "Q", "onto": "Q", "onto_to": "K"}], "immune": ["K"]}),
        True),
    "a contact that makes nobody a royal": (
        _bare([KING, QUEEN], contacts={"links": [{"mover": "Q", "onto": "Q", "onto_to": "Q"}]}),
        False),
    "contact links that are not a list": (_bare([KING, QUEEN], contacts={"links": "x"}), False),
    "a royal taken, the game running on, a hand filled by captures": (
        _bare([KING, {"id": "P"}], win="extinction", drops={"from_captures": True}), True),
    "the same with a drops block that says nothing else": (
        _bare([KING, {"id": "P"}], win="extinction", drops={}), True),
    "the same, but the hand is not filled by captures": (
        _bare([KING, {"id": "P"}], win="extinction", drops={"from_captures": False}), False),
    "a royal taken under checkmate cannot be taken": (
        _bare([KING, {"id": "P"}], win="checkmate", drops={"from_captures": True}), False),
    "a royal taken under royal capture ends the game": (
        _bare([KING, {"id": "P"}], win="royal_capture", drops={"from_captures": True}), False),
    "a piece promoted from a royal returns royal": (
        _bare([KING, {"id": "P", "promoted_from": "K"}], win="checkmate",
              drops={"from_captures": True}), True),
    "a piece promoted from an undeclared id returns as itself": (
        _bare([KING, {"id": "P", "promoted_from": "Z"}], win="checkmate",
              drops={"from_captures": True}), False),
    "a piece with no id beside a royal taken in a running game": (
        _bare([KING, {}], win="extinction", drops={}), True),
}


@pytest.mark.parametrize("label", sorted(SPELLING_CASES))
def test_the_spelling_level_read_branch_by_branch(label: str) -> None:
    spec, expected = SPELLING_CASES[label]
    assert royalty_can_multiply(spec) is expected
