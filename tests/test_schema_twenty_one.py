"""Schema 21 (Q843a): portal destinations as a map, and piece filters.

`docs/plan-portals-2.md` §2. ``rules.portals`` holds groups: a pair, written as before, or
a map of three to eight ends written with ``ends``, each leading to the end its ``to``
names (or nowhere: a way out only), with ``enter`` per end for whose pieces may go in and
``carries`` (on either spelling) for which pieces. One normaliser, `portal_groups`, reads
both spellings into one list of ends; the compiler emits ``dest`` and ``admits`` in place
of schema 19's ``partner`` and ``open_to``, and the core walks them.

`TestTheBumpMovedNothing` is the inertness proof: no mechanics hash moved, every perft of
the eight archived portal games is what the schema 20 core counted (`FROZEN_PERFT`, taken
on that core before the change), and the eight rows' tables fingerprints moved only by the
rename (`tests/test_schema_nine.py::FROZEN`'s Q843a comment). The ring, the funnel and the
filtered pair below are held against `tests/portal_oracle.py` over random walks.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from portal_oracle import PortalOracle
from retired_specs import REBUILT
from test_schema_nine import FROZEN
from test_schema_nineteen import (
    STEP,
    _core_portal_moves,
    _core_transits,
    _placed,
    _position,
    _unmoved,
    _with_portals,
)
from test_viz import _wrapped

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.spec.schema import (
    SCHEMA_VERSION,
    _schema_twenty_one_uses,
    portal_groups,
    portal_map_shape,
)
from chess_variant_sandbox.spec.transform import settle_schema_version
from chess_variant_sandbox.viz.wrapped import portal_payload

ROOT = Path(__file__).resolve().parents[1]
FAST_BUDGET_NODES = 150_000

# Perft of the eight archived portal games at depths 1 to 4, counted by the schema 20 core
# (main at cc91562f) before any of this change was built. The new core must count the same.
FROZEN_PERFT: dict[str, tuple[int, ...]] = {
    "carousel": (10, 101, 1315, 17088),
    "cylinder-wormhole": (20, 391, 9115, 208530),
    "divide": (33, 1082, 38583, 1359572),
    "ferry-no-straight-back": (28, 696, 17270, 407359),
    "ferry": (28, 712, 18109, 438193),
    "shut-wormhole": (5, 43, 284, 2837),
    "wormhole-adrift": (13, 155, 2097, 27455),
    "wormhole": (9, 74, 781, 7489),
}


def _perft_cases() -> list[Any]:
    cases = []
    for name, counts in FROZEN_PERFT.items():
        for depth, expected in enumerate(counts, start=1):
            marks = [pytest.mark.slow] if expected > FAST_BUDGET_NODES else []
            cases.append(pytest.param(name, depth, expected, marks=marks,
                                      id=f"{name}-{depth}"))
    return cases


def _schema(spec: dict[str, Any]) -> dict[str, Any]:
    spec["schema_version"] = 21
    return spec


def ring() -> dict[str, Any]:
    """Gardner with a ring of three on the empty third rank: a3 to c3 to e3 to a3. Either
    player may move the end on c3 a step, so the ring's shape changes as it is played."""
    return _schema(_with_portals(
        "gardner",
        [{"piece": "O", "ends": [
            {"at": "a3", "to": 1},
            {"at": "c3", "to": 2, "moved_by": [0, 1]},
            {"at": "e3", "to": 0},
        ]}],
        [STEP],
    ))


def funnel() -> dict[str, Any]:
    """Los Alamos with a funnel: b3 and e4 both lead to c4, a way out only, and the end on
    e4 is White's alone to go in by. Black may move the way out a step."""
    return _schema(_with_portals(
        "losalamos",
        [{"piece": "O", "ends": [
            {"at": "b3", "to": 2},
            {"at": "e4", "to": 2, "enter": [0]},
            {"at": "c4", "moved_by": [1]},
        ]}],
        [STEP],
    ))


def filtered() -> dict[str, Any]:
    """Los Alamos with a fixed pair on b3 and e4 that only knights may go in by at b3,
    and only rooks and pawns at e4."""
    return _schema(_with_portals(
        "losalamos",
        [{"piece": "O", "orange": "b3", "blue": "e4",
          "carries": {"orange": ["N"], "blue": ["R", "P"]}}],
    ))


GAMES = {"ring": ring, "funnel": funnel, "filtered": filtered}


class TestTheBumpMovedNothing:
    """No hash, no perft and no game moved; the fingerprints moved only by the rename."""

    @pytest.mark.parametrize(("name", "depth", "expected"), _perft_cases())
    def test_every_portal_games_perft_is_the_schema_twenty_cores(
        self, name: str, depth: int, expected: int,
    ) -> None:
        # `wormhole-adrift` left the library on 2026-10-12 (Q868) and is rebuilt from `wormhole`.
        spec = REBUILT[name]() if name in REBUILT else load_spec(ROOT / "archive" / f"{name}.json")
        engine = Engine(spec)
        assert engine.perft(engine.initial_state(), depth) == expected

    def test_every_hash_is_the_one_frozen(self) -> None:
        for path, (digest, _) in FROZEN.items():
            assert spec_hash(load_spec(ROOT / path)) == digest, path

    def test_only_the_showcases_use_schema_twenty_one(self) -> None:
        """Nothing written before the bump uses it: the users are exactly Q844b's showcases
        (`docs/reference/portal-showcase-ideas.md`), both in `archive/`."""
        assert sorted(
            path for path in FROZEN if _schema_twenty_one_uses(load_spec(ROOT / path))
        ) == ["archive/pawnproof-wormhole.json", "archive/sluice.json"]

    def test_a_game_without_portals_sends_no_portal_key(self) -> None:
        tables = canonical_tables(compile_spec(load_spec(ROOT / "rulesets" / "chess.json")))
        for key in ("portal_dest", "portal_group", "portal_admits"):
            assert key not in tables

    def test_a_pair_reads_as_two_ends_that_lead_to_each_other(self) -> None:
        spec = load_spec(ROOT / "archive" / "divide.json")
        (group,) = portal_groups(spec["rules"])
        assert portal_map_shape(group) == "pair"
        orange, blue = group["ends"]
        assert (orange["to"], blue["to"]) == (1, 0)
        assert (orange["enter"], blue["enter"]) == ({0}, {1})
        portals = compile_spec(spec).portals
        assert portals is not None
        first, second = portals.end_types
        assert (portals.dest[first], portals.dest[second]) == (second, first)
        assert [bool(mask) for mask in portals.admits[first]] == [True, False]


class TestTheVersionIsBumped:
    def test_the_schema_version_is_twenty_one_or_later(self) -> None:
        assert SCHEMA_VERSION >= 21

    def test_the_collector_names_each_key_whatever_its_value(self) -> None:
        assert _schema_twenty_one_uses(ring()) == {"rules.portals.ends", "rules.portals.ends.to"}
        assert _schema_twenty_one_uses(funnel()) == {
            "rules.portals.ends", "rules.portals.ends.to", "rules.portals.ends.enter",
        }
        assert _schema_twenty_one_uses(filtered()) == {"rules.portals.carries"}

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_an_older_stamp_is_refused_by_name(self, name: str) -> None:
        spec = GAMES[name]()
        spec["schema_version"] = 20
        assert any("added in schema 21" in problem for problem in validate_spec(spec))

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_each_test_game_validates_and_settles_to_twenty_one(self, name: str) -> None:
        spec = GAMES[name]()
        assert validate_spec(spec) == []
        spec["schema_version"] = SCHEMA_VERSION
        settle_schema_version(spec)
        assert spec["schema_version"] == 21


def _refused(spec: dict[str, Any], fragment: str) -> None:
    problems = validate_spec(spec)
    assert any(fragment in problem for problem in problems), problems


def _ring_with(**changes: Any) -> dict[str, Any]:
    """The ring with its end ``k`` replaced by ``changes[f"e{k}"]`` (``None`` drops it)."""
    spec = ring()
    ends = spec["rules"]["portals"][0]["ends"]
    for key, end in sorted(changes.items(), reverse=True):
        place = int(key[1:])
        if end is None:
            del ends[place]
        else:
            ends[place] = end
    return spec


class TestWhatTheValidatorRefuses:
    """`docs/plan-portals-2.md` §2.1's refusals, one each, and the filter's (§2.3)."""

    def test_a_destination_outside_the_group(self) -> None:
        _refused(_ring_with(e2={"at": "e3", "to": 3}), "a destination is always in its own group")

    def test_a_destination_that_is_the_end_itself(self) -> None:
        _refused(_ring_with(e2={"at": "e3", "to": 2}), "is the end itself")

    def test_an_end_nobody_can_use(self) -> None:
        spec = ring()
        spec["rules"]["portals"][0]["ends"].append({"at": "b3"})
        _refused(spec, "so nobody can use it")

    def test_a_mutual_pair_written_with_ends(self) -> None:
        _refused(_ring_with(e2=None, e0={"at": "a3", "to": 1}, e1={"at": "c3", "to": 0}),
                 "two ends are a pair")

    def test_a_one_way_pair_written_with_ends(self) -> None:
        _refused(_ring_with(e2=None, e0={"at": "a3", "to": 1}, e1={"at": "c3"}),
                 "two ends are a pair")

    def test_more_than_eight_ends(self) -> None:
        spec = _with_portals("chess", [
            {"piece": "O", "ends": [{"at": f"{f}4", "to": (i + 1) % 5}
                                    for i, f in enumerate("abcde")]},
            {"piece": "O", "ends": [{"at": f"{f}5", "to": (i + 1) % 4}
                                    for i, f in enumerate("abcd")]},
        ])
        _schema(spec)
        _refused(spec, "has 9 ends; at most 8")

    def test_more_than_four_groups(self) -> None:
        spec = _schema(_with_portals("chess", [
            {"piece": "O", "orange": f"{f}4", "blue": f"{f}5"} for f in "abcde"
        ]))
        _refused(spec, "holds 5 groups; at most 4")

    def test_one_destination_per_end(self) -> None:
        _refused(_ring_with(e0={"at": "a3", "to": [1, 2]}), "must be the index of another end")

    def test_one_destination_for_both_players(self) -> None:
        _refused(_ring_with(e0={"at": "a3", "to": {"0": 1, "1": 2}}),
                 "must be the index of another end")

    def test_enter_on_an_end_that_leads_nowhere(self) -> None:
        spec = funnel()
        spec["rules"]["portals"][0]["ends"][2]["enter"] = [0]
        _refused(spec, "but the end leads nowhere")

    def test_enter_naming_nobody(self) -> None:
        _refused(_ring_with(e0={"at": "a3", "to": 1, "enter": []}), "a way out only; write it")

    def test_two_spellings_in_one_group(self) -> None:
        spec = ring()
        spec["rules"]["portals"][0]["orange"] = "b3"
        _refused(spec, "unknown keys: orange")

    def test_two_ends_of_a_group_on_one_square(self) -> None:
        _refused(_ring_with(e2={"at": "a3", "to": 0}), "need squares of their own")

    def test_an_end_on_an_occupied_square(self) -> None:
        _refused(_ring_with(e2={"at": "e2", "to": 0}), "a square setup.placements puts a piece on")

    @pytest.mark.parametrize(("carries", "fragment"), [
        ([], "is empty, so no piece may go in"),
        (["N", "N"], "names a piece twice"),
        (["Z"], "which is not a declared piece"),
        (["O"], "a portal; a portal never goes in"),
        (["P", "N", "B", "R", "Q", "K"], "names every piece that can move"),
    ])
    def test_a_filter_that_says_nothing_or_names_the_wrong_thing(
        self, carries: list[str], fragment: str,
    ) -> None:
        spec = ring()
        spec["rules"]["portals"][0]["ends"][0]["carries"] = carries
        _refused(spec, fragment)

    def test_a_filter_on_an_end_nobody_goes_in_by(self) -> None:
        spec = funnel()
        spec["rules"]["portals"][0]["ends"][2]["carries"] = ["N"]
        _refused(spec, "but the end leads nowhere")
        spec = filtered()
        spec["rules"]["portals"][0]["passage"] = {"0": "blue_to_orange", "1": "blue_to_orange"}
        _refused(spec, "but nobody may go in by this end")

    def test_a_royal_may_be_filtered_in(self) -> None:
        spec = ring()
        spec["rules"]["portals"][0]["ends"][0]["carries"] = ["K"]
        assert validate_spec(spec) == []


class TestTheCompiledRule:
    def test_a_ring_leads_round_and_a_funnel_leads_in(self) -> None:
        for make, wanted in ((ring, [1, 2, 0]), (funnel, [2, 2, None])):
            spec = make()
            portals = compile_spec(spec).portals
            assert portals is not None
            first = portals.end_types[0]
            assert [
                None if portals.dest[t] < 0 else portals.dest[t] - first
                for t in portals.end_types
            ] == wanted
            assert portals.group_of == (0, 0, 0)
            assert portal_map_shape(portal_groups(spec["rules"])[0]) == make.__name__

    def test_admits_is_a_mask_of_declared_types_per_player(self) -> None:
        compiled = compile_spec(funnel())
        portals = compiled.portals
        assert portals is not None
        movers = {i for i, pid in enumerate(compiled.piece_ids[:portals.declared_types])
                  if pid != "O"}
        every = sum(1 << i for i in movers)
        b3, e4, c4 = portals.end_types
        assert portals.admits[b3] == (every, every)
        assert portals.admits[e4] == (every, 0)
        assert portals.admits[c4] == (0, 0)
        knights = compile_spec(filtered())
        assert knights.portals is not None
        orange = knights.portals.end_types[0]
        knight = 1 << knights.piece_ids.index("N")
        assert knights.portals.admits[orange] == (knight, knight)

    def test_the_end_ids_letter_a_map_by_place(self) -> None:
        assert compile_spec(ring()).piece_ids[-3:] == ("1a", "1b", "1c")
        assert compile_spec(filtered()).piece_ids[-2:] == ("1o", "1b")

    def test_a_piece_every_door_keeps_out_stays_one_way(self) -> None:
        """§2.3: a pawn kept out of every door keeps its one-way moves, so its push still
        clears the repetition window; a pawn some door admits does not."""
        spec = filtered()
        compiled = compile_spec(spec)
        assert compiled.piece_ids.index("P") not in compiled.one_way_types
        spec["rules"]["portals"][0]["carries"] = {"orange": ["N"], "blue": ["R"]}
        compiled = compile_spec(spec)
        assert compiled.piece_ids.index("P") in compiled.one_way_types

    def test_the_page_is_sent_each_ends_destination_and_filter(self) -> None:
        payload = portal_payload(compile_spec(funnel()))
        assert payload is not None
        b3, e4, c4 = payload["ends"]
        assert (b3["id"], b3["end"], b3["pair"]) == ("1a", "a", 1)
        assert (b3["dest"], e4["dest"], c4["dest"]) == (c4["type"], c4["type"], None)
        assert e4["open"] == [True, False] and c4["open"] == [False, False]
        assert e4["admits"][1] == [] and "O" not in e4["admits"][0]
        pair = portal_payload(compile_spec(filtered()))
        assert pair is not None
        assert [end["end"] for end in pair["ends"]] == ["orange", "blue"]
        assert pair["ends"][0]["admits"] == [["N"], ["N"]]


class TestTheOracleAgreesOnRandomWalks:
    """The core's transits, portal moves and checks through a portal equal the oracle's."""

    @pytest.mark.parametrize("name", sorted(GAMES))
    def test_random_walks(self, name: str) -> None:
        spec = GAMES[name]()
        engine = Engine(spec)
        oracle = PortalOracle(spec)
        rng = random.Random(21)
        seen = 0
        for _ in range(4):
            state = engine.initial_state()
            for _ in range(60):
                if engine.terminal(state):
                    break
                position = _position(engine, state)
                player = state.side_to_move
                transits = oracle.transits(position, player, _unmoved(state))
                assert _core_transits(engine, state) == transits
                assert _core_portal_moves(engine, state) == oracle.portal_moves(position, player)
                seen += len(transits)
                legal = engine.legal_moves(state)
                for move in legal:
                    after = engine.apply(state, move)
                    king = next(
                        (sq for sq, (pid, owner) in _position(engine, after).items()
                         if pid == "K" and owner == player),
                        None,
                    )
                    if king is not None:
                        assert not oracle.attacked_through(
                            _position(engine, after), 1 - player, king, _unmoved(after)
                        ), engine.describe_move(state, move)
                state = engine.apply(state, rng.choice(legal))
                assert state.key == engine._compute_hash(state)
        assert seen > 0


# A hand-built position ignores the setup, but the validator reads it: a king and a pawn a
# side, clear of every portal square, keep the pawn's promotion a rule that can fire.
_KING_AND_PAWN = [{"piece": "K", "squares": ["a1"]}, {"piece": "P", "squares": ["f2"]}]


def _bare_funnel() -> dict[str, Any]:
    spec = funnel()
    spec["setup"]["placements"] = _KING_AND_PAWN
    return spec


class TestTheMapAndTheFilterInPlay:
    """Hand-built positions: a ring goes round, a funnel comes out of its way out, and a
    filter that refuses a piece makes the end an obstacle to it, in transit and in check."""

    def _ring(self, placement: dict[str, tuple[str, Any]], side: int = 0) -> tuple[Engine, Any]:
        base = {"a3": ("1a", WILD), "c3": ("1b", WILD), "e3": ("1c", WILD),
                "a1": ("K", 0), "a5": ("K", 1)}
        return _placed(ring(), {**base, **placement}, side)

    def test_a_ring_goes_round(self) -> None:
        engine, state = self._ring({"c1": ("R", 0), "e1": ("R", 0)})
        texts = engine.move_texts(state)
        # Up the c-file into c3 comes out above e3; up the e-file into e3 above a3.
        assert "Rc1-c3>e4" in texts and "Re1-e3>a4" in texts
        assert not [t for t in texts if t.startswith("Rc1-c3>c4")]

    def test_a_funnel_comes_out_of_its_way_out_and_nothing_goes_in_there(self) -> None:
        engine, state = _placed(_bare_funnel(), {
            "b3": ("1a", WILD), "e4": ("1b", WILD), "c4": ("1c", WILD),
            "a1": ("K", 0), "f6": ("K", 1), "b1": ("R", 0), "c1": ("R", 0),
        })
        texts = engine.move_texts(state)
        assert "Rb1-b3>c5" in texts
        # The way out is a solid block: the rook up the c-file stops below it.
        assert "Rc1-c3" in texts and not [t for t in texts if t.startswith("Rc1-c4")]

    def test_an_end_closed_to_a_player_is_an_obstacle_to_that_player(self) -> None:
        engine, state = _placed(_bare_funnel(), {
            "b3": ("1a", WILD), "e4": ("1b", WILD), "c4": ("1c", WILD),
            "a1": ("K", 0), "f6": ("K", 1), "e1": ("R", 0), "e6": ("R", 1),
        }, side=1)
        texts = engine.move_texts(state)
        assert not [t for t in texts if t.startswith("Re6-e4")]
        assert "Re6-e5" in texts

    def _filtered(self, placement: dict[str, tuple[str, Any]], side: int = 0,
                  ) -> tuple[Engine, Any]:
        spec = filtered()
        spec["setup"]["placements"] = _KING_AND_PAWN
        base = {"b3": ("1o", WILD), "e4": ("1b", WILD)}
        return _placed(spec, {**base, **placement}, side)

    def test_a_filtered_end_lets_in_what_it_carries(self) -> None:
        engine, state = self._filtered({"a1": ("K", 0), "f6": ("K", 1), "c1": ("N", 0)})
        assert "Nc1-b3>d5" in engine.move_texts(state)

    def test_a_filtered_end_is_an_obstacle_to_what_it_refuses(self) -> None:
        engine, state = self._filtered({"a1": ("K", 0), "f6": ("K", 1), "b1": ("R", 0)})
        texts = engine.move_texts(state)
        assert not [t for t in texts if t.startswith("Rb1-b3")]
        assert "Rb1-b2" in texts and "Rb1-b4" not in texts

    def test_a_refused_piece_gives_no_check_through_the_end(self) -> None:
        # A rook up the b-file would come out of e4 onto e5, the black king -- but the end
        # on b3 carries only knights.
        engine, state = self._filtered({"a1": ("K", 0), "e5": ("K", 1), "b1": ("R", 0)}, 1)
        assert not engine.in_check(state, 1)

    def test_an_admitted_piece_gives_check_through_the_end(self) -> None:
        # A knight from c1 lands on b3 travelling up and to the left, and comes out of e4
        # one step up and to the left: d5, the black king.
        engine, state = self._filtered({"a1": ("K", 0), "d5": ("K", 1), "c1": ("N", 0)}, 1)
        assert engine.in_check(state, 1)
        oracle = PortalOracle(filtered())
        assert oracle.attacked_through(_position(engine, state), 0, engine._square_index("d5"))

    def test_a_filter_on_the_far_end_does_not_stop_a_piece_coming_out(self) -> None:
        # Into e4 by a rook (admitted there) comes out beside b3, which carries only knights.
        engine, state = self._filtered({"a1": ("K", 0), "f6": ("K", 1), "e1": ("R", 0)})
        assert "Re1-e4>b4" in engine.move_texts(state)



class TestTheWordsSayWhereEachEndLeads:
    """The page's minimal words for a map (`plan-portals-2.md` §5.1): by square, *leads
    to*; the look (the beak, the group map, dashed arms) is Q843b's."""

    @staticmethod
    def _payload(make: Any) -> Any:
        return json.dumps(portal_payload(compile_spec(make())))

    def _words(self, make: Any, index: int, square: str, there: str | None,
               sources: list[str]) -> Any:
        portals = self._payload(make)
        return _wrapped(
            f"(() => {{ const p = {portals}; const end = p.ends[{index}];"
            f" return portalWords(end, portalPartner(p, end), '{square}', 1,"
            f" s => ['white', 'black'][s], {json.dumps(there)}, {json.dumps(sources)}); }})()"
        )

    def test_a_way_out_says_nothing_goes_in_and_where_pieces_come_from(self) -> None:
        said = self._words(funnel, 2, "c4", None, ["b3", "e4"])
        assert said["label"] == (
            "The portal on c4. Nothing goes in here: pieces come out here from b3 and e4."
            " Only black may move it, as a turn.")

    def test_an_end_of_a_ring_says_where_it_leads_and_what_leads_to_it(self) -> None:
        said = self._words(ring, 0, "a3", "c3", ["e3"])
        assert said["label"] == (
            "The portal on a3. Both sides' pieces go in here and come out beside c3. Pieces"
            " come out here from e3. Nobody moves it.")
        assert said["says"].endswith("one step beyond the portal it leads to, in the direction"
                                     " it was going.")

    def test_a_one_player_end_of_a_funnel_says_whose(self) -> None:
        said = self._words(funnel, 1, "e4", "c4", [])
        assert said["label"].startswith(
            "The portal on e4. White's pieces go in here and come out beside c4.")

    def test_a_filtered_pair_keeps_its_words_and_says_its_filter(self) -> None:
        said = self._words(filtered, 0, "b3", "e4", [])
        assert said["label"] == (
            "The portal on b3. Both sides' pieces go in here or at its partner on e4, and"
            " come out beside the other. Nobody moves it. Only N may go in here.")

    def test_a_partner_is_the_end_it_leads_to_and_sources_lead_here(self) -> None:
        portals = self._payload(funnel)
        found = _wrapped(
            f"(() => {{ const p = {portals}; return [p.ends.map(e => (portalPartner(p, e)"
            f" || {{type: null}}).type), portalSources(p, p.ends[2]).map(e => e.type)]; }})()"
        )
        payload = portal_payload(compile_spec(funnel()))
        assert payload is not None
        b3, e4, c4 = (end["type"] for end in payload["ends"])
        assert found == [[c4, c4, None], [b3, e4]]
