"""F10c (Q752c): the core's own refusals at the boundary, each reached and read by its words.

`docs/plan-tests.md` F10c. The Rust core checks what crosses into it a second time, after
`engine/bridge.py` and the agents' resolvers have already refused anything malformed: a
table of the wrong shape (`rust/src/tables.rs::from_py`, `portal_tables`, `fold_tables`), a
call that does not fit the game (`lib.rs`: a round asked of a game that alternates, a seat
or square out of range, a spy game asked in the wrong phase), a player that does not fit
(`agents.rs::from_parts`/`check_against`, `eval.rs::PieceValues::check`) and a learned net's
store (`net.rs::Net::parse`). The bridge never sends any of these, so no other test reaches
them; they are reached here from Python, as Q752b found they could be, by handing
`_core.Core` a dict the bridge wrote and then corrupted, or by calling the core directly
with an argument the Python side would never pass.

Each family is one parametrised test, an arm per refusal, and every arm asserts the
`ValueError`'s own words, so a refusal that stops firing, fires for another reason or
changes its diagnosis fails its arm. The unmodified tables load (`test_the_tables_*`), so
each refusal is the corruption's and nothing else's. The tables are built once per spec
and copied shallowly per arm: an arm replaces a key, never mutates a list in place.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from test_schema_twenty_four import _gardner_folds

from chess_variant_sandbox import _core
from chess_variant_sandbox.agents import native_spec
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import MAX_PORTAL_ENDS

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent

Edit = Callable[[dict[str, Any]], dict[str, Any]]


def _chess() -> dict[str, Any]:
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / "chess.json").read_text(encoding="utf-8"))
    return spec


@cache
def _tables(name: str) -> dict[str, Any]:
    spec = _chess() if name == "chess" else _gardner_folds()
    return build_tables(compile_spec(spec))


def _refused(name: str, edit: dict[str, Any] | Edit, words: str) -> None:
    tables = dict(_tables(name))
    tables.update(edit(tables) if callable(edit) else edit)
    with pytest.raises(ValueError, match=words):
        _core.Core(tables)


def test_the_tables_every_arm_corrupts_load_as_written() -> None:
    for name in ("chess", "folds"):
        _core.Core(dict(_tables(name)))


# -- tables.rs: a table that does not fit the board, or a rule the core does not know --------

# Chess's table sizes: 6 types, 64 squares, 2 frames.
SIM = {"sim_rule": [0, 0, 0, 0], "sim_collision": [0] * 36, "zobrist_sim": [1] * 129}
DROPS = {"has_drops": True, "drops_regions": [-1] * (2 * 6 * 64)}
STRIKE = {"has_strike": True, "strike_to": [-1] * 6}
GOAL = [1] + [0] * 127

CHESS_ARMS: dict[str, tuple[dict[str, Any] | Edit, str]] = {
    "too-many-types": ({"n_types": 99}, r"^99 piece types; the core is built for at most 24"),
    "attack-dirs": ({"attack_dirs": [0]}, r"^attack_dirs holds 1 bytes for \d+ attack sources$"),
    "attack-sweep": ({"attack_sweep": [0]}, r"^attack_sweep holds 1 bytes for \d+ attack sources$"),
    "win": ({"win": "stalemate"}, r'^unknown win condition "stalemate"$'),
    "stalemate": ({"stalemate": "replay"}, r'^unknown stalemate rule "replay"$'),
    "checkmate-result": ({"checkmate_result": "draw"}, r'^unknown checkmate result "draw"$'),
    "strike-position": ({**STRIKE, "strike_position": 9}, r'^unknown strike position "9"$'),
    "strike-landing": (
        {**STRIKE, "strike_position": 3, "strike_landing_index": [0]},
        r'^unknown strike landing index length "1"$',
    ),
    "strike-removed-in-place": (
        {**STRIKE, "strike_position": 1, "strike_to": [-2] * 6},
        r'^unknown strike entry "removed in place outside stay"$',
    ),
    "strike-rebirth": (
        {**STRIKE, "strike_position": 0, "strike_rebirth": [0]},
        r'^unknown strike rebirth length "1"$',
    ),
    "strike-flips": (
        {**STRIKE, "strike_position": 0, "contact_mover_flips": [0]},
        r'^unknown contact mover flips length "1"$',
    ),
    "drops-faces-count": (
        {**DROPS, "drops_faces": [-1] * 5}, r"^drops_faces has 5 entries for 6 piece types$",
    ),
    "drops-face-off-the-types": (
        {**DROPS, "drops_faces": [6, -1, -1, -1, -1, -1]},
        r"^drops_faces pairs piece type 6 with more than one hand slot$",
    ),
    "drops-face-twice": (
        {**DROPS, "drops_faces": [0, 0, 0, -1, -1, -1]},
        r"^drops_faces pairs piece type 0 with more than one hand slot$",
    ),
    "goal-distances": (
        {"has_reach": True, "reach_squares": GOAL, "goal_distance_each": [0]},
        r"^goal_distance_each is not one table per goal square$",
    ),
    "extinction-sides": (
        {"has_extinction": True, "extinction_types": [0], "extinction_side_types": [[0]]},
        r"^extinction_side_types has 1 lists; it is empty or one per player$",
    ),
    "wild-beside-portals": (
        {"has_npcs": True, "has_portals": True},
        r"^wild pieces and portals share the third owner's block",
    ),
    "wild-tables": (
        {"has_npcs": True}, r"^wild-piece tables hold 0 and 0 squares, not the 64 this board",
    ),
    "entry-promotion": (
        {"entry_promotion": [0]},
        r"^the entry-promotion table holds 1 slots, not the 768 that 2 players x",
    ),
    "byte-two-owners": (
        {"n_frames": 44}, r"^6 piece types at 22 facings for two players need 265 board",
    ),
    "byte-three-owners": (
        {"n_frames": 30, "has_npcs": True, "npc_next": [-1] * 64, "npc_bounty": [0.0] * 64},
        r"^6 piece types at 15 facings for two players and a wild third owner need 271 board",
    ),
    "sweep-tables": (
        {"attack_path_modes": [0]},
        r"^compiled sweep tables hold 0 slots and 1 path modes, not the 0 and 0 this ruleset",
    ),
    "immobile-table": (
        {"immobile_squares": [0] * 767},
        r"^compiled dedup/immobile tables hold 768 and 767 entries, not the 768 that",
    ),
    "attack-index": (
        lambda t: {"attack_index": t["attack_index"][:-1]},
        r"^the attack table holds 128 slots, not the 129 that",
    ),
    "simultaneous": (
        {**SIM, "sim_rule": [0, 0, 0]},
        r"^the simultaneous tables do not fit this ruleset's types and squares$",
    ),
    "simultaneous-keys": (
        {**SIM, "zobrist_sim": [1] * 128},
        r"^the simultaneous tables do not fit this ruleset's types and squares$",
    ),
    "chase-rest": (
        {**SIM, "sim_chase": [1, 0, 0]}, r"^the chase switches do not fit this core",
    ),
    "chase-length": ({**SIM, "sim_chase": [0, 0]}, r"^the chase switches do not fit this core"),
    "speeds": ({**SIM, "sim_speed": [0]}, r"^the speeds do not fit this ruleset's types$"),
    "rest-keys-short": (
        {"piece_rest": [1] * 6, "zobrist_rest": []},
        r"^the rest tables do not fit this ruleset's types and squares$",
    ),
    "rest-over-max": (
        {"piece_rest": [5] * 6, "zobrist_rest": [1] * (5 * 128)},
        r"^the rest tables do not fit this ruleset's types and squares$",
    ),
    "rest-none": (
        {"piece_rest": [0] * 6, "zobrist_rest": []},
        r"^the rest tables do not fit this ruleset's types and squares$",
    ),
    "rest-keys-alone": ({"zobrist_rest": [1]}, r"^rest keys without a rest$"),
    "sight-flags": (
        {"sight_flags": [0]},
        r"^the sight tables do not fit this ruleset's squares \(or sit beside portals\)$",
    ),
    "sight-off-the-board": (
        {"sight_flags": [0, 0, 0], "sight_index": [0] * 768 + [1],
         "sight_srcs": [64, 1, 1, 0, 0]},
        r"^the sight tables do not fit this ruleset's squares",
    ),
    "sight-command-alone": ({"sight_command": True}, r"^sight sources without the sight's flags$"),
    "sight-index-alone": ({"sight_index": [0]}, r"^sight sources without the sight's flags$"),
    "spy-one-list": (
        {**SIM, "spy_types": [[0]]},
        r"^the spy tables are two lists of this ruleset's types, beside rules.simultaneous$",
    ),
    "spy-alternating": (
        {"spy_types": [[0], [0]]}, r"^the spy tables are two lists of this ruleset's types",
    ),
    "spy-deal-alone": ({"spy_deal": True}, r"^spy_deal without a spy$"),
}


@pytest.mark.parametrize("arm", CHESS_ARMS)
def test_a_table_that_does_not_fit_is_refused_by_name(arm: str) -> None:
    """`tables.rs::from_py`, on chess's tables with one key corrupted: a length one short or
    long, a rule word it does not know, a flag without the tables it needs, a board byte
    that would overflow."""
    edit, words = CHESS_ARMS[arm]
    _refused("chess", edit, words)


def _replace(key: str, index: int, value: Any) -> Edit:
    def edit(tables: dict[str, Any]) -> dict[str, Any]:
        row = list(tables[key])
        row[index] = value
        return {key: row}
    return edit


def _end_types(tables: dict[str, Any]) -> list[int]:
    return [t for t, group in enumerate(tables["portal_group"]) if group >= 0]


FOLD_ARMS: dict[str, tuple[dict[str, Any] | Edit, str]] = {
    "rows": ({"portal_movers": [0]}, r"^portal tables are not one row per each of \d+ types$"),
    "no-ends": (
        lambda t: {"portal_group": [-1] * len(t["portal_group"])},
        r"^portal ends are not two to eight contiguous types$",
    ),
    "dest-from-a-piece": (
        lambda t: _replace("portal_dest", 0, _end_types(t)[0])(t),
        r"^portal_dest or portal_admits names a type that is not an end$",
    ),
    "admits-on-a-piece": (
        lambda t: _replace("portal_admits", 0, 1)(t),
        r"^portal_dest or portal_admits names a type that is not an end$",
    ),
    "exits": ({"portal_exits": [-1]}, r"^portal_exits holds 1 entries$"),
    "no-back-rows": ({"portal_no_back": [0]}, r"^portal_no_back holds 1 rows for \d+ types$"),
    "bar-keys": ({"zobrist_portal_bar": [1]}, r"^zobrist_portal_bar holds 1 keys$"),
    "leap-dirs": (
        lambda t: {"leap_dirs": [*t["leap_dirs"], 0]},
        r"^leap_dirs holds (\d+) masks for (?!\1)\d+ landings$",
    ),
    "fold-kind": ({"portal_kind": [1]}, r"^portal_kind or fold_starts is not one row per type or end$"),
    "fold-start": (_replace("fold_starts", 0, -1), r"^fold_starts\[0\] does not match portal_kind$"),
    "fold-off-the-board": (
        _replace("fold_starts", 1, 25), r"^fold_starts\[1\] does not match portal_kind$",
    ),
    "fold-keys": ({"zobrist_fold": [1]}, r"^the fold tables do not fit the shapes they index$"),
}


@pytest.mark.parametrize("arm", FOLD_ARMS)
def test_a_portal_table_that_does_not_fit_is_refused_by_name(arm: str) -> None:
    """`tables.rs::portal_tables` and `fold_tables`, on a fold pair's tables (Gardner, 5 x 5,
    `test_schema_twenty_four._gardner_folds`) with one key corrupted."""
    edit, words = FOLD_ARMS[arm]
    _refused("folds", edit, words)


def test_the_core_holds_as_many_portal_ends_as_the_validator_allows() -> None:
    """CS7 (R15): `tables.rs::MAX_PORTAL_ENDS` sizes every by-end array in the core (a fold's
    squares, a transit's ends) and is the ceiling `portal_tables` refuses past; the validator
    allows `spec/schema.py::MAX_PORTAL_ENDS`. Equal, or a spec the validator passes is one the
    core refuses (or the core carries width nobody can use)."""
    assert _core.MAX_PORTAL_ENDS == MAX_PORTAL_ENDS == 8


# -- lib.rs: a call that does not fit the game ------------------------------------------------


@cache
def _engine(name: str) -> Engine:
    if name == "chess":
        return Engine(_chess())
    spec: dict[str, Any] = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    return Engine(spec)


def _spy_position(white: dict[str, str], black: dict[str, str]) -> GameState:
    engine = _engine("_spy_small")
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    return engine.make_state(placement)


def _agent(kind: str = "greedy", **fields: Any) -> tuple[Any, ...]:
    """``kind``'s core tuple with the named fields replaced (`agents.rs::from_parts`'
    order)."""
    names = ("kind", "nodes", "mobility", "max_depth", "rollout_depth", "exploration", "goal",
             "weights", "values", "immune", "net", "repairs")
    base = native_spec(kind)
    assert base is not None
    row = dict(zip(names, base, strict=True))
    row.update(fields)
    return tuple(row[name] for name in names)


def _call(case: str) -> None:
    chess = _engine("chess")
    opening = _pack(chess.initial_state())
    values = [1.0] * 6
    if case == "legal_moves_for":
        chess.core.legal_moves_for(opening, 0)
    elif case == "apply_round":
        chess.core.apply_round(opening, 0, 0)
    elif case == "select_round":
        chess.core.select_round(opening, 0, _agent("random"), values)
    elif case == "rest_turn":
        chess.core.rest_turn(opening, 2)
    elif case == "sight_of":
        chess.core.sight_of(opening, 64)
    elif case == "regret_solve-empty":
        _core.regret_solve([1.0], 0, 1)
    elif case == "regret_solve-short":
        _core.regret_solve([1.0, 2.0, 3.0], 2, 2)
    else:
        _spy_call(case)


# White holds its spy, Black has lost its: a round in which Black announces (spy.rs::phase).
ANNOUNCED = ({"c1": "K", "c2": "S"}, {"c5": "K", "a5": "R"})
BLIND = ({"c1": "K"}, {"c5": "K", "a5": "R"})


def _spy_call(case: str) -> None:
    spy = _engine("_spy_small")
    values = [1.0] * 5
    rounds = _engine("_retired_gardner_in_rounds")
    if case == "seat":
        rounds.core.select_round(_pack(rounds.initial_state()), 2, _agent(), [1.0] * 6)
    elif case == "announced-in-rounds":
        rounds.core.select_round(_pack(rounds.initial_state()), 0, _agent(), [1.0] * 6, 0, 1)
    elif case == "spy-turn-as-round":
        spy.core.select_round(_pack(spy.initial_state()), 0, _agent(), values)
    elif case == "answer-unannounced":
        spy.core.select_round(_pack(_spy_position(*ANNOUNCED)), 0, _agent(), values)
    elif case == "announcer-told":
        state = _spy_position(*ANNOUNCED)
        shown = spy.legal_moves_for(state, 1)[0]
        spy.core.select_round(_pack(state), 1, _agent(), values, 0, shown)
    elif case == "blind-told":
        state = _spy_position(*BLIND)
        shown = spy.legal_moves_for(state, 1)[0]
        spy.core.select_round(_pack(state), 0, _agent(), values, 0, shown)
    elif case == "blind-as-turn":
        spy.core.select_move(_pack(_spy_position(*BLIND)), _agent(), values)
    elif case == "announced-depth-as-turn":
        spy.core.select_move_depth(_pack(_spy_position(*ANNOUNCED)), _agent(), values)
    else:
        raise AssertionError(case)


CALL_ARMS = {
    "legal_moves_for": r"^legal_moves_for is for a simultaneous game; this one alternates$",
    "apply_round": r"^apply_round is for a simultaneous game; this one alternates$",
    "select_round": r"^select_round is for a simultaneous game; this one alternates$",
    "rest_turn": r"^a player is 0 or 1$",
    "sight_of": r"^square off the board$",
    "regret_solve-empty": r"^a 0 by 1 matrix has 0 entries, not 1$",
    "regret_solve-short": r"^a 2 by 2 matrix has 4 entries, not 3$",
    "seat": r"^a seat is 0 or 1$",
    "announced-in-rounds": r"^only an announced round of a spy game has an announced move$",
    "spy-turn-as-round": (
        r"^this spy game is in its alternating phase, played one move at a time; ask select_move$"
    ),
    "answer-unannounced": r"^the answerer's choice in an announced round needs the announced move$",
    "announcer-told": r"^the announcer chooses before any move is announced; leave announced out$",
    "blind-told": r"^only an announced round of a spy game has an announced move$",
    "blind-as-turn": r"^this spy game is in its blind phase, played in rounds; ask select_round$",
    "announced-depth-as-turn": (
        r"^this spy game is in its announced phase, played in rounds; ask select_round$"
    ),
}


@pytest.mark.parametrize("case", CALL_ARMS)
def test_a_call_that_does_not_fit_the_game_is_refused_by_name(case: str) -> None:
    """`lib.rs`: a round's calls on a game that alternates, a player, seat or square out of
    range, a solver's matrix of the wrong size, and each wrong question of a spy game's
    phase (`tests/_spy_small.json`; `_retired_gardner_in_rounds.json` for rounds without a
    spy)."""
    with pytest.raises(ValueError, match=CALL_ARMS[case]):
        _call(case)


def test_the_spy_positions_are_in_the_phases_the_arms_name() -> None:
    """The arms above read their phase off these positions; the core's own `phase` says so."""
    spy = _engine("_spy_small")
    assert spy.phase(spy.initial_state()).name == "alternating"
    assert spy.phase(_spy_position(*ANNOUNCED)).name == "announced"
    assert spy.phase(_spy_position(*BLIND)).name == "blind"


# -- agents.rs, eval.rs, net.rs: a player that does not fit -----------------------------------


def _store(n_inputs: int = 1, accumulator: int = 1, hidden: tuple[int, int] = (1, 1),
           version: int = 1, magic: bytes = b"CVSNET01", tail: bytes = b"") -> bytes:
    """A `CVSNET01` store laid out as `net.rs::Net::parse` reads it, all weights zero."""
    h1, h2 = hidden
    head = (bytes(h1 * accumulator) + bytes(4 * h1) + bytes(h2 * h1) + bytes(4 * h2)
            + bytes(h2) + bytes(4))
    body = bytes(2 * n_inputs * accumulator) + bytes(2 * accumulator) + head + head
    return magic + struct.pack("<5I", version, n_inputs, accumulator, h1, h2) + body + tail


WEIGHTS = (0.0,) * len(VOCABULARY)

AGENT_ARMS: dict[str, tuple[tuple[Any, ...], str]] = {
    "repairs": (
        _agent("minimax:nodes=50", repairs=1 << 31),
        r"^search repairs 0x80000000 name a bit outside 0x\w+; a player spelling a repair",
    ),
    "net-for-random": (
        _agent("random", net=_store()), r'^"random" has no evaluator for a net to replace$',
    ),
    "net-for-mcts": (
        _agent("mcts:nodes=50", net=_store()), r'^"mcts" has no evaluator for a net to replace$',
    ),
    "net-beside-weights": (
        _agent(weights=WEIGHTS, net=_store()),
        r"^a net was given beside a weight vector or piece values",
    ),
    "net-beside-values": (
        _agent(values=(1.0,) * 6, net=_store()),
        r"^a net was given beside a weight vector or piece values",
    ),
    "values-alone": (
        _agent(values=(0.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
        r"^piece values were given with no weight vector, which names the built-in objective",
    ),
    "values-all-nothing": (
        _agent(weights=WEIGHTS, values=(0.0,) * 6),
        r"^every non-royal piece is worth nothing, which is not an opinion about",
    ),
    "net-magic": (
        _agent(net=_store(magic=b"CVSNET00")),
        r"^the net store does not start with the CVSNET01 magic$",
    ),
    "net-layout": (
        _agent(net=_store(version=2)),
        r"^the net store was written for input layout 2 and this core reads layout 1; it is",
    ),
    "net-accumulator-none": (
        _agent(net=_store(accumulator=0)),
        r"^the net's accumulator width 0 is outside 1..=2048$",
    ),
    "net-accumulator-wide": (
        _agent(net=_store(n_inputs=0, accumulator=2049)),
        r"^the net's accumulator width 2049 is outside 1..=2048$",
    ),
    "net-first-hidden": (
        _agent(net=_store(hidden=(0, 1))), r"^the net's first hidden width 0 is outside 1..=128$",
    ),
    "net-second-hidden": (
        _agent(net=_store(hidden=(1, 129))),
        r"^the net's second hidden width 129 is outside 1..=128$",
    ),
    "net-tail": (
        _agent(net=_store(tail=b"\x00\x00\x00")),
        r"^the net store has 3 bytes after its last layer$",
    ),
}


@pytest.mark.parametrize("arm", AGENT_ARMS)
def test_a_player_that_does_not_fit_is_refused_by_name(arm: str) -> None:
    """`agents.rs::AgentSpec::from_parts` and `check_against`, `eval.rs::PieceValues::check`
    and `net.rs::Net::parse`, asked through `select_move` on chess with a core tuple no
    resolver writes: a repair bit the core does not define, a net beside a player with no
    evaluator or beside a heuristic, piece values with no weights or worth nothing, and a
    store with each of its header fields and its length wrong."""
    agent, words = AGENT_ARMS[arm]
    chess = _engine("chess")
    with pytest.raises(ValueError, match=words):
        chess.core.select_move(_pack(chess.initial_state()), agent, [1.0] * 6)


def test_the_store_the_arms_corrupt_parses() -> None:
    """The well-formed store gets past the parser, to the check that it was laid out for
    this game's inputs (which a one-input net is not): the arms above are each its one
    corruption."""
    chess = _engine("chess")
    with pytest.raises(ValueError) as refused:
        chess.core.select_move(_pack(chess.initial_state()), _agent(net=_store()), [1.0] * 6)
    assert not re.search(r"net store|width", str(refused.value)), refused.value
