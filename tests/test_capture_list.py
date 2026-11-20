"""R-L4b stage 2 (`docs/reference/capture-effects-design.md` §6): every entry on every taking.

A capture list in its own order is handed to the compiler past the validator
(`compile.with_capture_list`; the validator speaks the old keys until stage 3) and the core walks
it (`Game::walk_general`). The declared oracle `tests/capture_oracle.py` plays the same list on
the same board in Python; here the two agree on every list of up to three entries over the fuzz
parents (`fuzz_lifted.generate.parents`), each given a blast, a strike, a capture hand and a
bounty so that every entry has a rule to walk. Beside the oracle: `make` and `unmake` hold the
position together (perft against an `apply` count), the incremental key holds, and the legality
filter plays the list as `make` does (no legal move leaves the mover in check).
"""

from __future__ import annotations

import copy
import itertools
import random
from typing import Any
from unittest import mock

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.compile import (
    CAPTURE_KINDS,
    CompiledSpec,
    compile_spec,
    with_capture_list,
)
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_PROMOTION,
    MOVE_PUSH,
    MOVE_SWEEP,
    move_from,
    move_gate,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fuzz_lifted.generate import parents
from chess_variant_sandbox.spec.schema import SpecError
from tests import capture_oracle
from tests.engine_hold_specs import library

POSITIONS = ("stay", "swap", "home")


def _with_every_rule(spec: dict[str, Any], position: str, sweep: bool = False) -> dict[str, Any]:
    """``spec`` with a blast, a strike demoting down the value ladder (placed ``position``), a
    capture hand and a bounty on every piece but the royals, counted in points. Under ``sweep``
    the most valuable piece sweeps a rook's lines three squares (so the list reads a path), and
    the strike switches the capturer's side (Andernach), so the mover half has two things to
    write."""
    spec = copy.deepcopy(spec)
    pieces = spec["pieces"]
    royals = [p["id"] for p in pieces if p.get("royal")]
    others = sorted((p for p in pieces if not p.get("royal")), key=lambda p: -p.get("value", 1))
    if sweep:
        others[0]["moves"] = [{"kind": "sweep", "vectors": [[1, 0]], "symmetry": "rotate4",
                               "mode": "both", "range": 3}]
    chain = [{"from": a["id"], "to": b["id"]} for a, b in itertools.pairwise(others)]
    for piece in others:
        piece["bounty"] = max(1, int(piece.get("value", 1)))
    rules = spec.setdefault("rules", {})
    rules["scoring"] = {"count": "points"}
    rules["explode_on_capture"] = {"radius": 1, "immune": royals}
    rules["strike"] = {"effect": "demote", "position": position, "immune": royals, "chain": chain}
    if sweep:
        rules["strike"]["switches_sides"] = [p["id"] for p in others]
    if position == "choose":
        rules["strike"]["landing"] = {"vectors": LANDINGS}
    rules.setdefault("drops", {})["from_captures"] = True
    return spec


#: A `choose` table's landings: the four orthogonal neighbours of the struck piece.
LANDINGS = [[0, 1], [0, -1], [1, 0], [-1, 0]]


def _past_validator(spec: dict[str, Any], capture: list[dict[str, Any]] | None = None) -> CompiledSpec:
    """``spec`` compiled past the validator (the pairs the list orders are refused there until
    stage 3), with ``capture`` handed to the compile (`compile_spec`'s tests-only list)."""
    with mock.patch.object(compile_module, "validate_or_raise", lambda spec: spec):
        return compile_spec(spec, capture)


def _compiled(name: str, position: str, sweep: bool = False) -> CompiledSpec:
    """The parent with every rule, compiled past the validator (which refuses a strike beside a
    hand and a blast, the pairs the list exists to order)."""
    spec = dict(parents())[name]
    with mock.patch.object(compile_module, "validate_or_raise", lambda spec: spec):
        return compile_spec(_with_every_rule(spec, position, sweep))


def _is_capture(state: GameState, move: int) -> bool:
    return move_type(move) in (MOVE_EN_PASSANT, MOVE_SWEEP) or state.board[move_to(move)] != 0


def _agree(engine: Engine, state: GameState) -> int:
    """Every capture from ``state`` played by the core and by the oracle; the count checked."""
    checked = 0
    for move in engine.legal_moves(state):
        if not _is_capture(state, move):
            continue
        try:
            want = capture_oracle.play(
                engine.spec, state.board, state.side_to_move, move, state.ep_victim,
                state.hands, state.points, state.purse,
            )
        except capture_oracle.Unsupported:
            continue
        after = engine.apply(state, move)
        got = (list(after.board), [list(h) for h in after.hands], list(after.points), list(after.purse))
        assert got == (want.board, want.hands, want.points, want.purse), (
            engine.describe_move(state, move), [e.do for e in engine.spec.capture or ()]
        )
        checked += 1
    return checked


def _walk(engine: Engine, seed: int, plies: int) -> int:
    """A seeded playout, the oracle asked at every position on the way; the captures checked."""
    rng = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        checked += _agree(engine, state)
        moves = engine.legal_moves(state)
        captures = [m for m in moves if _is_capture(state, m)]
        state = engine.apply(state, rng.choice(captures if captures and rng.random() < 0.6 else moves))
    return checked


def _lists(longest: int, more: tuple[dict[str, Any], ...] = ()) -> list[list[dict[str, Any]]]:
    kinds = [{"do": kind} for kind in CAPTURE_KINDS]
    kinds += [{"do": "blast", "spares_taken": True}, {"do": "blast", "on": ["landing", "blast"]},
              {"do": "strike", "on": ["landing", "blast", "path"]}, *more]
    return [list(combo) for n in range(1, longest + 1) for combo in itertools.product(kinds, repeat=n)]


class TestTheListPlaysAsTheOracleSays:
    @pytest.mark.parametrize("name", [name for name, _ in parents()])
    def test_every_list_of_one_or_two_entries(self, name: str) -> None:
        checked = 0
        for position in POSITIONS:
            compiled = _compiled(name, position)
            assert capture_oracle.supported(with_capture_list(compiled, [])), name
            for index, entries in enumerate(_lists(2)):
                engine = Engine(with_capture_list(compiled, entries))
                checked += _walk(engine, seed=index, plies=6)
        assert checked > 0

    @pytest.mark.parametrize("name", [name for name, _ in parents()])
    def test_a_sweeper_and_a_switch(self, name: str) -> None:
        """A sweep's path read by each entry, and a side switch in the mover half."""
        checked = 0
        for position in ("stay", "home"):
            compiled = _compiled(name, position, sweep=True)
            for index, entries in enumerate(_lists(2)):
                engine = Engine(with_capture_list(compiled, entries))
                checked += _walk(engine, seed=index, plies=8)
        assert checked > 0

    PUSH_FORMS = (
        {"do": "strike", "on": ["landing", "push"]},
        {"do": "contacts", "on": ["landing", "push"]},
        {"do": "blast", "on": ["landing", "push"], "spread": True},
    )

    @staticmethod
    def _push_and_two_tables(name: str) -> CompiledSpec:
        """The parent with every rule, a push on every non-royal piece, and T3's contact table
        beside its strike (placed `home`, so a pushed-off piece has somewhere to go)."""
        spec = dict(parents())[name]
        return _past_validator(_changing_contacts(_with_push(_with_every_rule(spec, "home"))))

    @pytest.mark.parametrize("name", [name for name, _ in parents()])
    def test_a_push_and_two_tables_every_list_of_one_or_two(self, name: str) -> None:
        compiled = self._push_and_two_tables(name)
        assert compiled.contacts is not None and capture_oracle.supported(with_capture_list(compiled, []))
        checked = 0
        for index, entries in enumerate(_lists(2, self.PUSH_FORMS)):
            checked += _walk(Engine(with_capture_list(compiled, entries)), seed=index, plies=6)
        assert checked > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("name", [name for name, _ in parents()])
    def test_a_push_and_two_tables_every_list_of_up_to_three(self, name: str) -> None:
        compiled = self._push_and_two_tables(name)
        checked = 0
        for index, entries in enumerate(_lists(3, self.PUSH_FORMS)):
            checked += _walk(Engine(with_capture_list(compiled, entries)), seed=index, plies=12)
        assert checked > 0

    @pytest.mark.slow
    def test_a_gating_hand_every_list_of_up_to_two(self) -> None:
        compiled = _past_validator(_with_every_rule(library("seirawan"), "stay"))
        checked = 0
        for index, entries in enumerate(_lists(2)):
            checked += _walk(Engine(with_capture_list(compiled, entries)), seed=index, plies=24)
        assert checked > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("name", [name for name, _ in parents()])
    def test_every_list_of_up_to_three_entries(self, name: str) -> None:
        checked = 0
        for position, sweep in itertools.product(POSITIONS, (False, True)):
            compiled = _compiled(name, position, sweep)
            for index, entries in enumerate(_lists(3)):
                engine = Engine(with_capture_list(compiled, entries))
                checked += _walk(engine, seed=index, plies=16)
        assert checked > 0


def _row(entries: list[dict[str, Any]], seen: Any, position: str = "stay",
         sweep: bool = False) -> None:
    """One of §3.1's rows: the list ``entries`` plays as the oracle says on every parent, and
    the case the row refused (``seen(engine, before, move, after)``) arises at least once."""
    arose = checked = 0
    for index, (name, _) in enumerate(parents()):
        compiled = _compiled(name, position, sweep)
        listed = entries(compiled) if callable(entries) else entries
        engine = Engine(with_capture_list(compiled, listed))
        checked += _walk(engine, seed=index, plies=12)
        rng = random.Random(index)
        state = engine.initial_state()
        for _ in range(30):
            if engine.terminal(state) is not None:
                break
            moves = engine.legal_moves(state)
            for move in moves:
                if _is_capture(state, move) and seen(engine, state, move, engine.apply(state, move)):
                    arose += 1
            captures = [m for m in moves if _is_capture(state, m)]
            state = engine.apply(state, rng.choice(captures or moves))
    assert checked > 0 and arose > 0, (checked, arose)


def _hand_grew(before: GameState, after: GameState) -> bool:
    return sum(map(sum, after.hands)) > sum(map(sum, before.hands))


def _blasted(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
    """The taker went up: neither its origin nor its target holds it."""
    from chess_variant_sandbox.engine.moves import move_from
    return after.board[move_from(move)] == 0 and after.board[move_to(move)] != before.board[move_from(move)]


class TestTheRowsTheListLifts:
    """§3.1's rows, each by name: the pair plays under a list, as the oracle says. The rest of
    §3.1 and §3.2's four are `TestTheRestOfTheRows` below."""

    def test_T1_position_a_spared_piece_is_struck_after_its_blast(self) -> None:
        def seen(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            target = move_to(move)
            return (_blasted(engine, before, move, after) and before.board[target] != 0
                    and after.board[target] != 0)
        for position in POSITIONS:
            _row([{"do": "blast", "spares_taken": True}, {"do": "strike"}],
                 seen if position == "stay" else (lambda *a: True), position)

    def test_T2_T5_H4_a_struck_piece_stands_or_is_pocketed_whichever_comes_first(self) -> None:
        def pocketed(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return _hand_grew(before, after)

        def stands(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return not _hand_grew(before, after) and after.board[move_to(move)] != 0
        _row([{"do": "hand"}, {"do": "strike"}], pocketed)
        _row([{"do": "strike"}, {"do": "hand"}], stands)

    def test_T4_switch_a_gone_capturer_switches_nothing(self) -> None:
        _row([{"do": "blast"}, {"do": "strike"}], _blasted, sweep=True)

    def test_H5_the_hand_takes_the_victim_and_the_switch_the_capturer(self) -> None:
        def seen(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            landed = after.board[move_to(move)]
            return (_hand_grew(before, after) and landed != 0
                    and engine.spec.owner_of(landed) != before.side_to_move)
        _row([{"do": "hand"}, {"do": "contacts"}], seen, position="home", sweep=True)

    def test_W1_disagree_by_names_the_sweepers_whose_path_is_struck(self) -> None:
        def swept(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return move_type(move) == MOVE_SWEEP
        # `by` naming a royal, which never sweeps: the path is read by nobody.
        _row(lambda c: [{"do": "strike", "on": ["landing", "path"],
                         "by": [c.piece_ids[c.royal_types[0]]]}], swept, sweep=True)
        _row([{"do": "strike", "on": ["landing", "path"]}], swept, sweep=True)

    def test_W3_both_a_sweeps_blasts_and_strikes_follow_the_list(self) -> None:
        def swept(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return move_type(move) == MOVE_SWEEP
        _row([{"do": "blast", "on": ["path"], "spread": True},
              {"do": "strike", "on": ["landing", "path", "blast"]}], swept, sweep=True)
        _row([{"do": "strike", "on": ["landing", "path"]}, {"do": "blast", "on": ["path"]}],
             swept, sweep=True)


class TestTheCoreHoldsTheListTogether:
    """`unmake` takes the walk back (perft against an `apply` count, which never unmakes), the
    key follows every write, and the filter plays the list as `make` does."""

    LISTS = (
        [{"do": "strike"}, {"do": "blast"}, {"do": "hand"}, {"do": "pay"}],
        [{"do": "blast", "spares_taken": True}, {"do": "strike", "on": ["landing", "blast"]}],
        [{"do": "hand"}, {"do": "strike"}, {"do": "blast", "on": ["landing", "blast"]}],
    )

    @staticmethod
    def _count(engine: Engine, state: GameState, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(
            TestTheCoreHoldsTheListTogether._count(engine, engine.apply(state, m), depth - 1)
            for m in engine.legal_moves(state)
        )

    @pytest.mark.parametrize("entries", LISTS)
    @pytest.mark.parametrize("name", ["five", "gardner"])
    def test_perft_and_the_key(self, name: str, entries: list[dict[str, Any]]) -> None:
        checks = dict(parents())[name]["rules"]["win"] == "checkmate"
        engine = Engine(with_capture_list(_compiled(name, "stay"), entries))
        rng = random.Random(7)
        state = engine.initial_state()
        for _ in range(14):
            if engine.terminal(state) is not None:
                break
            assert engine.perft(state, 2) == self._count(engine, state, 2)
            mover = state.side_to_move
            for move in engine.legal_moves(state):
                after = engine.apply(state, move)
                assert engine.hash(after) == engine._compute_hash(after)
                if checks:
                    # The filter played the list as `make` does: no legal move leaves the
                    # mover's royal attacked (one that takes the other royal wins at once).
                    assert not engine.in_check(after, mover) or engine.terminal(after) is not None
            moves = engine.legal_moves(state)
            captures = [m for m in moves if _is_capture(state, m)]
            state = engine.apply(state, rng.choice(captures or moves))


class TestTheListIsHandedOverByTestsOnly:
    def test_no_parent_carries_a_list_and_a_lifted_pair_does(self) -> None:
        """Stage 3: a game the old keys play as they always did crosses no list; old keys
        stating a pair only the list plays (here a strike that keeps its victim beside a
        capture hand, T2) hand the core their default-order list."""
        for name, spec in parents():
            assert compile_spec(spec).capture is None
            assert [e.do for e in _compiled(name, "stay").capture or ()] == [
                "pay", "hand", "blast", "strike"]

    def test_a_kind_needs_its_rule(self) -> None:
        bare = compile_spec(dict(parents())["five"])
        with pytest.raises(SpecError):
            with_capture_list(bare, [{"do": "blast"}])
        with pytest.raises(SpecError):
            with_capture_list(bare, [{"do": "strike"}])

    def test_spread_is_a_blasts(self) -> None:
        with pytest.raises(SpecError):
            with_capture_list(_compiled("five", "stay"), [{"do": "strike", "spread": True}])

    def test_an_empty_list_is_plain_capture(self) -> None:
        # The blast stays declared: handed to the compile, an empty list leaves the royal its
        # captures (`_without_self_destroying_captures` asks the list, §2.5).
        spec = _with_every_rule(dict(parents())["five"], "stay")
        plain = Engine(_past_validator(spec, []))
        bare = Engine(compile_spec(dict(parents())["five"]))
        state, other = plain.initial_state(), bare.initial_state()
        rng = random.Random(3)
        for _ in range(20):
            if plain.terminal(state) is not None:
                break
            moves = plain.legal_moves(state)
            assert sorted(moves) == sorted(bare.legal_moves(other))
            move = rng.choice(moves)
            state, other = plain.apply(state, move), bare.apply(other, move)
            assert state.board == other.board


# --- The rest of §3.1 and §3.2's rows, and §2.5's readers (R-L4b2b). ---

_LANDING_BITS = 0x1F << 25


def _others(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The non-royal pieces, most valuable first (`_with_every_rule`'s order)."""
    return sorted((p for p in spec["pieces"] if not p.get("royal")), key=lambda p: -p.get("value", 1))


def _states(engine: Engine, seed: int, plies: int, prefer: Any = None) -> Any:
    """The positions of a seeded playout that prefers captures (and ``prefer``'s moves)."""
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(plies):
        if engine.terminal(state) is not None:
            return
        yield state
        moves = engine.legal_moves(state)
        favoured = [m for m in moves if prefer is not None and prefer(state, m)]
        captures = [m for m in moves if _is_capture(state, m)]
        pool = favoured or (captures if captures and rng.random() < 0.7 else moves)
        state = engine.apply(state, rng.choice(pool))


def _fanned(engine: Engine, state: GameState) -> int:
    """How many captures from ``state`` are offered once per chosen square (two words or more
    that differ only in the landing)."""
    words: dict[int, int] = {}
    for move in engine.legal_moves(state):
        if _is_capture(state, move):
            words[move & ~_LANDING_BITS] = words.get(move & ~_LANDING_BITS, 0) + 1
    return sum(1 for count in words.values() if count > 1)


def _contacts_choose(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` (from `_with_every_rule`) with its strike restated as a contact table placed
    ``choose``: every non-royal piece demotes the one it takes down the value ladder."""
    spec = copy.deepcopy(spec)
    others = _others(spec)
    rules = spec["rules"]
    del rules["strike"]
    down = {a["id"]: b["id"] for a, b in itertools.pairwise(others)}
    rules["contacts"] = {
        "position": "choose", "landing": {"vectors": LANDINGS},
        "links": [{"mover": m["id"], "onto": o, "onto_to": to}
                  for m in others for o, to in down.items()],
    }
    return spec


def _with_push(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with a king's-step push on every non-royal piece (one piece a line, thrown off
    the edge)."""
    spec = copy.deepcopy(spec)
    for piece in _others(spec):
        piece["moves"].append({"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                               "mode": "move", "push": True})
    return spec


def _changing_contacts(spec: dict[str, Any]) -> dict[str, Any]:
    """T3 and U9's contact table: the least valuable non-royal piece becomes the most valuable
    one when it takes any non-royal piece (a link that changes the mover, and no victim)."""
    spec = copy.deepcopy(spec)
    others = _others(spec)
    low, high = others[-1]["id"], others[0]["id"]
    spec["rules"]["contacts"] = {"links": [{"mover": low, "onto": o["id"], "mover_to": high}
                                           for o in others]}
    return spec


def _with_upkeep(spec: dict[str, Any]) -> dict[str, Any]:
    """``spec`` with every piece's move costing one coin a row (`rules.purse.upkeep`)."""
    spec = copy.deepcopy(spec)
    for piece in spec["pieces"]:
        piece["upkeep"] = 1
    spec["rules"]["purse"] = {"start": 60, "upkeep": True}
    spec["rules"].pop("drops", None)  # a drop's cost is refused beside upkeep (`_upkeep_holds`)
    return spec


def _arises(engine: Engine, seeds: int, plies: int, seen: Any, prefer: Any = None) -> tuple[int, int]:
    """Over ``seeds`` playouts: the captures the oracle checked, and how often ``seen(engine,
    before, move, after)`` held for a capture."""
    checked = arose = 0
    for seed in range(seeds):
        for state in _states(engine, seed, plies, prefer):
            checked += _agree(engine, state)
            for move in engine.legal_moves(state):
                if _is_capture(state, move) and seen(engine, state, move, engine.apply(state, move)):
                    arose += 1
    return checked, arose


class TestTheRestOfTheRows:
    """T1.choose, T4.choose, T3, the three push rows (§3.1), and §3.2's gating and upkeep
    rows, each by name: the pair plays under a list as the declared oracle says, and the case
    the row refused arises."""

    def test_T1_choose_a_spared_piece_is_offered_its_squares_and_a_destroyed_one_none(self) -> None:
        checked = fans_spared = fans_destroyed = 0
        for index, (name, _) in enumerate(parents()):
            compiled = _compiled(name, "choose")
            spared = Engine(with_capture_list(compiled, [{"do": "blast", "spares_taken": True}, {"do": "strike"}]))
            destroyed = Engine(with_capture_list(compiled, [{"do": "blast"}, {"do": "strike"}]))
            checked += _walk(spared, seed=index, plies=12)
            for state in _states(spared, index, 16):
                fans_spared += _fanned(spared, state)
                fans_destroyed += _fanned(destroyed, state)
        assert checked > 0 and fans_spared > 0
        assert fans_destroyed == 0, "a capture whose blast destroys its victim was offered landings"

    def test_T4_choose_a_contact_placed_choose_beside_a_blast(self) -> None:
        checked = fans_spared = fans_destroyed = 0
        for index, (_name, spec) in enumerate(parents()):
            compiled = _past_validator(_contacts_choose(_with_every_rule(spec, "stay")))
            spared = Engine(with_capture_list(compiled, [{"do": "blast", "spares_taken": True}, {"do": "contacts"}]))
            destroyed = Engine(with_capture_list(compiled, [{"do": "blast"}, {"do": "contacts"}]))
            checked += _walk(spared, seed=index, plies=12)
            for state in _states(spared, index, 16):
                fans_spared += _fanned(spared, state)
                fans_destroyed += _fanned(destroyed, state)
        assert checked > 0 and fans_spared > 0 and fans_destroyed == 0

    def test_T3_a_strike_and_a_contact_table_each_read_their_own(self) -> None:
        def both(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            # The strike placed the victim on the origin (swap), the contact changed the mover.
            high = engine.spec.piece_ids.index(_others(spec)[0]["id"])
            landed = after.board[move_to(move)]
            return (move_type(move) != MOVE_PROMOTION
                    and after.board[move_from(move)] != 0 and landed != 0
                    and engine.spec.type_of(landed) == high
                    and engine.spec.type_of(before.board[move_from(move)]) != high)
        checked = arose = 0
        for _name, parent in parents():
            spec = _changing_contacts(_with_every_rule(parent, "swap"))
            compiled = _past_validator(spec)
            assert compiled.contacts is not None and compiled.strike is not None
            for entries in ([{"do": "strike"}, {"do": "contacts"}], [{"do": "contacts"}, {"do": "strike"}]):
                engine = Engine(with_capture_list(compiled, entries))
                got = _arises(engine, 2, 12, both)
                checked, arose = checked + got[0], arose + got[1]
            # A list with no contacts entry reads the strike table alone: the mover never changes.
            strike_only = Engine(with_capture_list(compiled, [{"do": "strike"}]))
            assert strike_only.spec.contacts is None
            assert _arises(strike_only, 2, 12, both)[1] == 0
        assert checked > 0 and arose > 0

    @staticmethod
    def _push_row(entries: list[dict[str, Any]], position: str, seen: Any) -> None:
        checked = arose = 0
        for _name, parent in parents():
            compiled = _past_validator(_with_push(_with_every_rule(parent, position)))
            engine = Engine(with_capture_list(compiled, entries))
            spec = engine.spec

            def takes(state: GameState, move: int, spec: CompiledSpec = spec) -> bool:
                return move_type(move) == MOVE_PUSH and spec.push_taken(state.board, move) != 0
            got = _arises(engine, 4, 20, seen, prefer=takes)
            checked, arose = checked + got[0], arose + got[1]
        assert checked > 0 and arose > 0, (checked, arose)

    @staticmethod
    def _thrown_off_and_back(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
        """A push that took (a piece left the board) whose victim the list placed again."""
        pieces = sum(1 for v in before.board if v)
        return (move_type(move) == MOVE_PUSH and engine.spec.push_taken(before.board, move) != 0
                and sum(1 for v in after.board if v) == pieces)

    def test_push_X2_strike_a_pushed_off_piece_is_struck(self) -> None:
        self._push_row([{"do": "strike", "on": ["landing", "push"]}], "home", self._thrown_off_and_back)

    def test_push_X2_contacts_a_pushed_off_piece_meets_its_link(self) -> None:
        self._push_row([{"do": "contacts", "on": ["landing", "push"]}], "home", self._thrown_off_and_back)

    def test_push_X3_a_pushed_off_piece_sets_off_a_blast(self) -> None:
        def blasted(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return (move_type(move) == MOVE_PUSH and engine.spec.push_taken(before.board, move) != 0
                    and after.board[move_to(move)] == 0)
        self._push_row([{"do": "blast", "on": ["landing", "push"]}], "stay", blasted)
        self._push_row([{"do": "blast", "on": ["landing", "push"], "spread": True}], "stay", blasted)

    @staticmethod
    def _gated_capture(state: GameState, move: int) -> bool:
        return move_gate(move) != 0 and _is_capture(state, move)

    def _gating_row(self, entries: list[dict[str, Any]], seen: Any) -> None:
        compiled = _past_validator(_with_every_rule(library("seirawan"), "stay"))
        engine = Engine(with_capture_list(compiled, entries))
        checked, arose = _arises(engine, 6, 40, seen, prefer=self._gated_capture)
        assert checked > 0 and arose > 0, (checked, arose)
        # `unmake` takes the gate and the list back (perft against an `apply` count).
        for state in _states(engine, 1, 30, prefer=self._gated_capture):
            if any(self._gated_capture(state, m) for m in engine.legal_moves(state)):
                assert engine.perft(state, 2) == TestTheCoreHoldsTheListTogether._count(engine, state, 2)
                break

    def test_gating_blast_the_gated_piece_enters_before_the_blast(self) -> None:
        def seen(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return self._gated_capture(before, move) and after.board[move_to(move)] == 0
        self._gating_row([{"do": "blast"}], seen)

    def test_gating_captures_the_gate_spends_the_hand_and_the_capture_fills_it(self) -> None:
        def seen(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            side = before.side_to_move
            return (self._gated_capture(before, move) and after.board[move_from(move)] != 0
                    and sum(after.hands[side]) == sum(before.hands[side]))
        self._gating_row([{"do": "hand"}, {"do": "pay"}], seen)

    @staticmethod
    def _upkeep_row(spec_of: Any, entries: list[dict[str, Any]], seen: Any) -> None:
        checked = arose = 0
        for _name, parent in parents():
            compiled = _past_validator(spec_of(parent))
            assert compiled.purse is not None and compiled.purse.upkeep
            engine = Engine(with_capture_list(compiled, entries))
            got = _arises(engine, 2, 14, seen)
            checked, arose = checked + got[0], arose + got[1]
        assert checked > 0 and arose > 0, (checked, arose)

    def test_upkeep_U9_a_link_that_changes_the_mover_is_not_charged(self) -> None:
        def changed(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            landed, moving = after.board[move_to(move)], before.board[move_from(move)]
            return landed != 0 and engine.spec.type_of(landed) != engine.spec.type_of(moving)

        def spec_of(parent: dict[str, Any]) -> dict[str, Any]:
            spec = _with_every_rule(parent, "stay")
            del spec["rules"]["strike"]
            return _with_upkeep(_changing_contacts(spec))
        self._upkeep_row(spec_of, [{"do": "contacts"}], changed)

    def test_upkeep_U10_a_switched_capturer_is_charged_to_its_mover(self) -> None:
        def switched(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            landed = after.board[move_to(move)]
            return landed != 0 and engine.spec.owner_of(landed) != before.side_to_move

        def spec_of(parent: dict[str, Any]) -> dict[str, Any]:
            spec = _with_every_rule(parent, "home", sweep=True)
            return _with_upkeep(spec)
        self._upkeep_row(spec_of, [{"do": "strike"}], switched)

    def test_upkeep_a_shot_pays_for_its_origin_unless_a_blast_takes_it_first(self) -> None:
        def stayed(engine: Engine, before: GameState, move: int, after: GameState) -> bool:
            return after.board[move_from(move)] == before.board[move_from(move)]
        spec_of = lambda parent: _with_upkeep(_with_every_rule(parent, "stay"))  # noqa: E731
        self._upkeep_row(spec_of, [{"do": "strike"}], stayed)
        self._upkeep_row(spec_of, [{"do": "blast"}, {"do": "strike"}], _blasted)


class TestTheReadersAskTheList:
    """§2.5: `fans_out` (above, T1.choose and T4.choose), the own-link landings and
    `_without_self_destroying_captures` ask the list; the pin shortcut stays off under any
    declared blast, which a list's blast entries need (`CaptureProgram::from_order`)."""

    def test_a_hand_ahead_of_a_choosing_strike_offers_no_squares(self) -> None:
        compiled = _compiled("five", "choose")
        choosing = Engine(with_capture_list(compiled, [{"do": "strike"}]))
        pocketing = Engine(with_capture_list(compiled, [{"do": "hand"}, {"do": "strike"}]))
        fans = sum(_fanned(choosing, s) for s in _states(choosing, 0, 16))
        assert fans > 0
        assert sum(_fanned(pocketing, s) for s in _states(choosing, 0, 16)) == 0

    def test_an_own_link_is_offered_only_by_a_contacts_entry(self) -> None:
        spec = _with_every_rule(dict(parents())["five"], "stay")
        del spec["rules"]["strike"]
        others = _others(spec)
        spec["rules"]["contacts"] = {"links": [
            {"mover": a["id"], "onto": b["id"], "own": True, "mover_to": others[0]["id"]}
            for a in others for b in others if a is not b
        ]}
        compiled = _past_validator(spec)

        def own_landings(engine: Engine) -> int:
            count = 0
            for seed in range(5):
                for state in _states(engine, seed, 30):
                    side = state.side_to_move
                    count += sum(1 for m in engine.legal_moves(state)
                                 if state.board[move_to(m)] and move_type(m) != MOVE_PUSH
                                 and engine.spec.owner_of(state.board[move_to(m)]) == side)
            return count
        assert own_landings(Engine(with_capture_list(compiled, [{"do": "contacts"}]))) > 0
        assert own_landings(Engine(with_capture_list(compiled, [{"do": "blast"}, {"do": "hand"}]))) == 0
        checked = _walk(Engine(with_capture_list(compiled, [{"do": "blast"}, {"do": "contacts"}])), 0, 12)
        assert checked > 0

    def test_a_royal_keeps_its_captures_where_no_blast_reads_its_landing(self) -> None:
        spec = _with_every_rule(dict(parents())["five"], "stay")
        royal = spec["pieces"].index(next(p for p in spec["pieces"] if p.get("royal")))
        exploding = _past_validator(spec, [{"do": "blast"}])
        path_only = _past_validator(spec, [{"do": "blast", "on": ["path"]}])
        assert exploding.leaps != path_only.leaps
        assert path_only.leaps == _past_validator(spec, []).leaps
        assert exploding.attack_sources != path_only.attack_sources, royal
