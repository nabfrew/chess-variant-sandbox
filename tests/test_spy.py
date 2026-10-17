"""Schema 27's spy (Q935a, `docs/plan-spy.md` §1, §2, §4).

The phases and the reply turn on the plan's positions (§1.5, E1 to E12), the refusals
(§1.7), the core held to `tests/spy_oracle.py` step by step over random walks of the test
specs, the alternating phase move for move as its alternating twin's, `perft_spy` frozen
after the oracle agreed, and `random` playing every phase through the core and through the
Python loop alike.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import validate_spec
from tests import spy_oracle as oracle

HERE = Path(__file__).parent
SPECS = ("_spy_small", "_spy_blast", "_spy_stock", "_spy_resting")


def _load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


_ENGINES: dict[str, Engine] = {}


def _engine(name: str) -> Engine:
    if name not in _ENGINES:
        _ENGINES[name] = Engine(_load(name))
    return _ENGINES[name]


def _walk(engine: Engine, seed: int, limit: int = 200) -> list[tuple[GameState, tuple[int, ...]]]:
    """A random walk through the core, each step checked against the oracle: the phase,
    the legal moves, the position after, and the incremental key against the slow one."""
    rng = random.Random(seed)
    state = engine.initial_state()
    walked: list[tuple[GameState, tuple[int, ...]]] = []
    for _ in range(limit):
        if engine.terminal(state) is not None:
            break
        phase = engine.phase(state)
        assert (phase.name, phase.announcer) == oracle.phase(engine, state)
        if phase.is_round:
            lists = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
            assert lists == oracle.legal(engine, state)
            pair = rng.choice(lists[0]), rng.choice(lists[1])
            after = engine.apply_round(state, *pair)
            assert after == oracle.round_(engine, state, *pair)
            walked.append((state, pair))
        else:
            moves = engine.legal_moves(state)
            assert moves == oracle.legal(engine, state)
            move = rng.choice(moves)
            after = engine.apply(state, move)
            assert after == oracle.turn(engine, state, move)
            walked.append((state, (move,)))
        assert after.key == engine._compute_hash(after)
        state = after
    return walked


@pytest.mark.parametrize("name", SPECS)
def test_the_core_agrees_with_the_oracle_over_random_walks(name: str) -> None:
    engine = _engine(name)
    seen: set[str] = set()
    for seed in range(30):
        for state, _ in _walk(engine, seed):
            seen.add(engine.phase(state).name)
    # Every walk set reaches the rounds, so the seams are crossed, not only the turns.
    assert "alternating" in seen and ("announced" in seen or "blind" in seen), seen


# ------------------------------------------------------------------ the plan's positions


def _move(engine: Engine, state: GameState, origin: str, target: str, player: int | None = None) -> int:
    """The legal move from ``origin`` to ``target``: the side to move's on a turn, ``player``'s
    in a round."""
    moves = engine.legal_moves(state) if player is None else engine.legal_moves_for(state, player)
    a, b = engine._square_index(origin), engine._square_index(target)
    found = [m for m in moves if decode_move(m)[:2] == (a, b)]
    assert len(found) == 1, (origin, target, [engine.describe_move(state, m) for m in moves])
    return found[0]


def _position(engine: Engine, white: dict[str, str], black: dict[str, str], side: int = 0) -> GameState:
    placement = {square: (piece, 0) for square, piece in white.items()}
    placement.update({square: (piece, 1) for square, piece in black.items()})
    state = engine.make_state(placement, side_to_move=side)
    assert engine.terminal(state) is None
    return state


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[engine._square_index(square)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def _phase(engine: Engine, state: GameState) -> tuple[str, int]:
    phase = engine.phase(state)
    assert (phase.name, phase.announcer) == oracle.phase(engine, state)
    return phase.name, phase.announcer


def _blast(**simultaneous: Any) -> Engine:
    spec = _load("_spy_blast")
    spec["rules"]["simultaneous"] = dict(simultaneous)
    return Engine(spec)


class TestThePhases:
    def test_e1_a_spy_taken_by_the_first_player_owes_the_reply_turn(self) -> None:
        engine = _engine("_spy_small")
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
        assert _phase(engine, state) == ("alternating", -1)
        state = engine.apply(state, _move(engine, state, "a4", "c4"))
        assert _phase(engine, state) == ("reply", -1)
        # The reply is an ordinary turn: the king retakes the spy's taker.
        state = engine.apply(state, _move(engine, state, "d5", "c4"))
        assert _at(engine, state, "c4") == ("K", 1)
        assert _phase(engine, state) == ("announced", 1)
        # A round is refused one move at a time.
        with pytest.raises(ValueError, match="announced phase"):
            engine.apply(state, engine.legal_moves_for(state, 0)[0])

    def test_e1_without_the_reply_the_taker_would_escape(self) -> None:
        """What the reply turn prevents (§1.3): in a round the taker sees the recapture
        coming and steps away, and the king lands on an empty square."""
        engine = _engine("_spy_small")
        state = _position(engine, {"a1": "K", "c4": "R", "c2": "S"}, {"d5": "K"})
        assert _phase(engine, state) == ("announced", 1)
        king = _move(engine, state, "d5", "c4", player=1)
        away = _move(engine, state, "c4", "c5", player=0)
        after = engine.apply_round(state, away, king)
        assert _at(engine, after, "c4") == ("K", 1) and _at(engine, after, "c5") == ("R", 0)

    def test_e2_a_spy_taken_by_the_second_player_is_the_same_rule(self) -> None:
        engine = _engine("_spy_small")
        state = _position(engine, {"d1": "K", "c2": "S"}, {"a5": "K", "a2": "R", "c4": "S"}, side=1)
        with pytest.raises(ValueError, match="alternating phase"):
            engine.apply_round(state, engine.legal_moves_for(state, 0)[0], engine.legal_moves_for(state, 1)[0])
        state = engine.apply(state, _move(engine, state, "a2", "c2"))
        assert _phase(engine, state) == ("reply", -1) and state.side_to_move == 0
        state = engine.apply(state, _move(engine, state, "d1", "c2"))
        assert _phase(engine, state) == ("announced", 0) and state.side_to_move == 1
        # The side to move stops flipping once rounds begin.
        pair = engine.legal_moves_for(state, 0)[0], engine.legal_moves_for(state, 1)[0]
        after = engine.apply_round(state, *pair)
        assert after.side_to_move == 1 and _phase(engine, after) == ("announced", 0)

    def test_e3_a_spy_lost_on_its_owner_s_move_starts_rounds_at_once(self) -> None:
        engine = _engine("_spy_blast")
        state = _position(
            engine, {"e1": "K", "b2": "N", "c3": "S"}, {"d5": "K", "d3": "R", "a5": "S"},
        )
        state = engine.apply(state, _move(engine, state, "b2", "d3"))
        assert _at(engine, state, "c3") is None and _at(engine, state, "d3") is None
        assert _phase(engine, state) == ("announced", 0)

    def test_e4_both_spies_lost_in_one_move_is_a_blind_round_and_no_deal(self) -> None:
        engine = _blast(tie="initiative")
        state = _position(
            engine, {"e1": "K", "b2": "N", "c3": "S"}, {"d5": "K", "d3": "R", "e4": "S"},
        )
        assert state.initiative == 1
        state = engine.apply(state, _move(engine, state, "b2", "d3"))
        assert _phase(engine, state) == ("blind", -1)
        assert state.initiative == 1  # no announced round dealt it (E10)

    def test_e5_the_answerer_dodges_or_trades_the_spy(self) -> None:
        engine = _engine("_spy_small")
        state = _position(
            engine, {"e1": "K", "a5": "R", "c1": "S"}, {"e5": "K", "d5": "R", "b3": "N"},
        )
        assert _phase(engine, state) == ("announced", 1)
        strike = _move(engine, state, "b3", "c1", player=1)
        dodged = engine.apply_round(state, _move(engine, state, "c1", "b1", player=0), strike)
        assert _at(engine, dodged, "b1") == ("S", 0) and _at(engine, dodged, "c1") == ("N", 1)
        assert _phase(engine, dodged) == ("announced", 1)
        traded = engine.apply_round(state, _move(engine, state, "a5", "d5", player=0), strike)
        assert _at(engine, traded, "d5") == ("R", 0) and _at(engine, traded, "c1") == ("N", 1)
        assert _phase(engine, traded) == ("blind", -1)

    def test_e6_both_spies_never_fall_in_one_round(self) -> None:
        """A proof over §1.3's table: a round is played only where at most one spy stands,
        so no round can take two."""
        for has in ((True, True), (True, False), (False, True), (False, False)):
            for side in (0, 1):
                rounds = has == (False, False) or (has[0] != has[1] and has[side])
                if rounds:
                    assert sum(has) <= 1

    def test_e7_there_is_no_check_to_carry_across_the_switch(self) -> None:
        engine = _engine("_spy_small")
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"e4": "K", "c4": "S"})
        state = engine.apply(state, _move(engine, state, "a4", "c4"))
        assert _phase(engine, state) == ("reply", -1)
        # The rook on c4 attacks d4; the reply may step there all the same.
        _move(engine, state, "e4", "d4")
        fled = engine.apply(state, _move(engine, state, "e4", "e5"))
        assert _phase(engine, fled) == ("announced", 1)

    @pytest.mark.parametrize("rule", [
        {"castling": {"king": "K", "rook": "R", "king_steps": 2}},
        {"en_passant": True},
        {"win": "checkmate"},
    ])
    def test_e8_rules_rounds_refuse_are_refused_for_the_whole_game(self, rule: dict[str, Any]) -> None:
        spec = _load("_spy_small")
        spec["rules"].update(rule)
        assert any("rules.simultaneous beside" in error for error in validate_spec(spec))

    def test_e9_the_void_run_starts_empty_and_a_bar_carries_into_blind_rounds(self) -> None:
        engine = _engine("_spy_small")
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
        state = engine.apply(state, _move(engine, state, "a4", "c4"))
        state = engine.apply(state, _move(engine, state, "d5", "c4"))
        assert state.void_run == (-1, -1) and state.bars == ()
        # E5's position with a bar between the kings, which the round does not touch.
        state = _position(
            engine, {"e1": "K", "a5": "R", "c1": "S"}, {"e5": "K", "d5": "R", "b3": "N"},
        )
        bar = (engine._square_index("e1"), engine._square_index("e5"))
        state = state.with_(bars=(bar,))
        key = engine._compute_hash(state)
        state = state.with_(key=key, repetition=(key,))
        strike = _move(engine, state, "b3", "c1", player=1)
        after = engine.apply_round(state, _move(engine, state, "a5", "d5", player=0), strike)
        assert _phase(engine, after) == ("blind", -1) and after.bars == (bar,)

    def test_e10_the_initiative_is_dealt_to_the_announcer_once(self) -> None:
        engine = _engine("_spy_resting")
        state = _position(engine, {"d1": "K", "c2": "S"}, {"a4": "K", "a2": "R", "c4": "S"}, side=1)
        assert state.initiative == 1
        state = engine.apply(state, _move(engine, state, "a2", "c2"))
        assert state.initiative == 1  # the reply turn is not a round
        state = engine.apply(state, _move(engine, state, "d1", "c2"))
        assert _phase(engine, state) == ("announced", 0) and state.initiative == 0
        assert state.key == engine._compute_hash(state)
        assert state == oracle.turn(engine, oracle_reply(engine), _move(engine, oracle_reply(engine), "d1", "c2"))
        # A tie the announcer's token wins passes it on, and no later round deals it back.
        meet = engine.apply_round(
            state, _move(engine, state, "c2", "b3", player=0),
            _move(engine, state, "a4", "b3", player=1),
        )
        assert _at(engine, meet, "b3") == ("K", 0) and meet.initiative == 1
        assert _phase(engine, meet) == ("announced", 0)
        quiet = engine.apply_round(
            meet, engine.legal_moves_for(meet, 0)[0], engine.legal_moves_for(meet, 1)[0],
        )
        assert quiet.initiative == 1

    def test_e11_the_first_round_starts_with_nothing_resting(self) -> None:
        engine = _engine("_spy_resting")
        state = _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})
        state = engine.apply(state, _move(engine, state, "a4", "c4"))
        state = engine.apply(state, _move(engine, state, "d5", "c4"))
        assert state.rest == (-1, -1)
        # The king that made the reply may move again in round 1.
        king = _move(engine, state, "c4", "c5", player=1)
        after = engine.apply_round(state, _move(engine, state, "a1", "a2", player=0), king)
        assert after.rest == (engine._square_index("a2"), engine._square_index("c5"))

    def test_e12_a_reply_turn_ends_as_any_turn_and_a_round_as_any_round(self) -> None:
        engine = _engine("_spy_small")
        state = _position(engine, {"d3": "K", "a4": "R", "c2": "S"}, {"e4": "K", "c4": "S"})
        state = engine.apply(state, _move(engine, state, "a4", "c4"))
        # The spy's loss is a material change: the repetition window starts again.
        assert len(state.repetition) == 1
        taken = engine.apply(state, _move(engine, state, "e4", "d3"))
        outcome = engine.terminal(taken)
        assert outcome is not None and (outcome.winner, outcome.reason) == (1, "royal_captured")
        # Both royals taken in one announced round is a draw.
        state = _position(engine, {"c1": "K", "c4": "R", "a1": "S"}, {"c5": "K", "b1": "R"})
        assert _phase(engine, state) == ("announced", 1)
        both = engine.apply_round(
            state, _move(engine, state, "c4", "c5", player=0),
            _move(engine, state, "b1", "c1", player=1),
        )
        outcome = engine.terminal(both)
        assert outcome is not None and (outcome.winner, outcome.reason) == (None, "royal_captured")


def oracle_reply(engine: Engine) -> GameState:
    """E10's position after Black's capture of the white spy, played by the oracle."""
    state = _position(engine, {"d1": "K", "c2": "S"}, {"a4": "K", "a2": "R", "c4": "S"}, side=1)
    return oracle.turn(engine, state, _move(engine, state, "a2", "c2"))


# ------------------------------------------------------------------------ the refusals


def _refused(spec: dict[str, Any], words: str) -> None:
    errors = validate_spec(spec)
    assert any(words in error for error in errors), errors


class TestTheRefusals:
    def test_every_test_spec_validates(self) -> None:
        for name in SPECS:
            assert validate_spec(_load(name)) == []

    def test_false_written_out(self) -> None:
        spec = _load("_spy_small")
        spec["pieces"][2]["spy"] = False
        _refused(spec, "pieces[2].spy false is the default")

    def test_a_value_that_is_not_true(self) -> None:
        spec = _load("_spy_small")
        spec["pieces"][4]["spy"] = "yes"
        _refused(spec, "pieces[4].spy must be true")

    def test_without_rounds(self) -> None:
        spec = _load("_spy_small")
        del spec["rules"]["simultaneous"]
        _refused(spec, "a spy decides when a game plays in rounds; declare rules.simultaneous")

    def test_on_a_royal(self) -> None:
        spec = _load("_spy_small")
        spec["pieces"][0]["spy"] = True
        _refused(spec, "losing a royal ends the game, so its loss could never start a round")

    def test_a_player_starting_without_one(self) -> None:
        spec = _load("_spy_small")
        spec["setup"]["mirror"] = False
        placements = [dict(p, player=0) for p in spec["setup"]["placements"]]
        placements += [
            {"piece": p["piece"], "player": 1, "squares": [s[0] + str(6 - int(s[1])) for s in p["squares"]]}
            for p in spec["setup"]["placements"] if p["piece"] != "S"
        ]
        spec["setup"]["placements"] = placements
        _refused(spec, "player 1 starts with no spy on the board, so the game would begin in rounds")

    def test_nothing_can_remove_it(self) -> None:
        spec = _load("_spy_small")
        for piece in spec["pieces"]:
            if piece["id"] != "S":
                piece["captures"] = [pid for pid in ("K", "R", "N", "P")]
        _refused(spec, "nothing can remove player 0's spy")

    def test_a_promotion_to_it(self) -> None:
        spec = _load("_spy_small")
        spec["rules"]["promotion"]["to"] = ["R", "S"]
        _refused(spec, "rules.promotion turns a piece into the spy 'S'")

    def test_a_stock_holding_it(self) -> None:
        spec = _load("_spy_stock")
        spec["rules"]["drops"]["stock"] = {"N": 1, "S": 1}
        _refused(spec, "rules.drops.stock puts the spy 'S' in a hand")

    def test_a_hand_filled_from_captures(self) -> None:
        spec = _load("_spy_stock")
        del spec["rules"]["drops"]["from_captures"]
        _refused(spec, "a taken spy would come back as its taker's")

    def test_a_strike_that_switches_it(self) -> None:
        spec = _load("_spy_small")
        spec["pieces"][4]["captures"] = ["P"]
        spec["pieces"][4]["moves"][0]["mode"] = "both"
        spec["rules"]["strike"] = {"switches_sides": ["S"]}
        _refused(spec, "rules.strike.switches_sides names the spy 'S'")

    def test_a_strike_chain_into_it(self) -> None:
        spec = _load("_spy_small")
        spec["rules"]["strike"] = {"chain": [{"from": "N", "to": "S"}]}
        _refused(spec, "rules.strike.chain turns a piece into the spy 'S'")

    def test_extinction_counting_it_on_its_own(self) -> None:
        spec = _load("_spy_small")
        spec["rules"]["win"] = "none"
        spec["pieces"][0].pop("royal")
        spec["rules"]["extinction"] = {"pieces": ["K", "S"], "per_type": True}
        _refused(spec, "losing the spy would end the game, so its rounds would never be played")

    def test_extinction_summed_is_allowed(self) -> None:
        spec = _load("_spy_small")
        spec["rules"]["extinction"] = {"pieces": ["R", "S"], "min_count": 1}
        assert not [e for e in validate_spec(spec) if "spy" in e]

    def test_a_wild_spy(self) -> None:
        spec = _load("_spy_small")
        spec["rules"]["npcs"] = [{"piece": "S", "route": ["a3", "b3", "c3", "d3", "e3"]}]
        assert any("wild piece" in error for error in validate_spec(spec))

    def test_an_older_version_may_not_say_spy(self) -> None:
        spec = _load("_spy_small")
        spec["schema_version"] = 26
        _refused(spec, "the spec uses pieces.spy, added in schema 27")


# ---------------------------------------------------------- compiled, crossed, inert


def test_the_collector_names_the_spy() -> None:
    from chess_variant_sandbox.spec.schema import _schema_twenty_seven_uses

    assert "pieces.spy" in _schema_twenty_seven_uses(_load("_spy_small"))
    assert "pieces.spy" not in _schema_twenty_seven_uses(_load("_simultaneous_resting"))


def test_each_side_s_spy_types_come_from_the_opening_board() -> None:
    spec = _engine("_spy_small").spec
    index = spec.piece_ids.index("S")
    assert spec.spy is not None and spec.spy.types == ((index,), (index,))
    assert spec.simultaneous is not None and not spec.simultaneous.deal_initiative
    resting = _engine("_spy_resting").spec
    assert resting.simultaneous is not None and resting.simultaneous.deal_initiative


def test_two_spy_types_are_one_summed_count() -> None:
    """§1.2: a side given two spies has lost "the spy" when it has none left, so a second
    spy type standing keeps the game alternating (`Game::has_spy`, not `short_of`)."""
    spec = _load("_spy_small")
    spec["pieces"].append(dict(copy.deepcopy(spec["pieces"][4]), id="T", name="second spy"))
    spec["setup"]["placements"][3]["squares"].remove("a2")
    spec["setup"]["placements"].append({"piece": "T", "squares": ["a2"]})
    assert validate_spec(spec) == []
    engine = Engine(spec)
    assert engine.spec.spy is not None and len(engine.spec.spy.types[0]) == 2
    state = _position(
        engine, {"a1": "K", "a4": "R", "c2": "S", "e2": "T"}, {"d5": "K", "c4": "S", "e4": "T"},
    )
    state = engine.apply(state, _move(engine, state, "a4", "c4"))
    assert _phase(engine, state) == ("alternating", -1)  # Black still has its T
    state = engine.apply(state, _move(engine, state, "d5", "e5"))
    state = engine.apply(state, _move(engine, state, "c4", "e4"))
    assert _phase(engine, state) == ("reply", -1)  # and now none


def test_the_boundary_sends_the_spy_only_where_there_is_one() -> None:
    from chess_variant_sandbox.engine.bridge import _INERT_DEFAULTS, build_tables

    tables = build_tables(_engine("_spy_small").spec)
    assert tables["spy_types"] and tables["spy_deal"] is False
    assert build_tables(_engine("_spy_resting").spec)["spy_deal"] is True
    plain = build_tables(Engine(_load("_simultaneous_resting")).spec)
    assert plain["spy_types"] == _INERT_DEFAULTS["spy_types"]
    assert plain["spy_deal"] == _INERT_DEFAULTS["spy_deal"]


def test_phase_is_refused_without_a_spy() -> None:
    engine = Engine(_load("_simultaneous_resting"))
    with pytest.raises(ValueError, match="phase is for a spy game"):
        engine.phase(engine.initial_state())


# A tournament of random players on the two games every bump holds still, its records
# (duration left out) hashed: taken on main's core 838a4ae05c743155 before Q935a and on this
# one, and equal (`docs/plan-spy.md` §2.3). Random play is movegen and the seat streams, so
# only a change to either moves it, and that change owes its perfts too.
_INERT_TOURNAMENTS = {
    "rulesets/chess.json": "5a0fe3f6b91534ab6556db450c59dd6a075af82ca19dda0f7f6e6b70c41bfbc1",
    "archive/gardner-in-rounds.json":
        "79e4c2b3f899124b04381fcde8f4c035851b5deddf4981f9ef24773fecaa8fe6",
}


@pytest.mark.parametrize("path", sorted(_INERT_TOURNAMENTS))
def test_a_game_without_a_spy_plays_byte_for_byte_as_before(path: str) -> None:
    import hashlib

    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

    spec = load_spec(HERE.parent / path)
    result = run_tournament(
        spec, TournamentConfig(agents=("random",), games_per_pairing=8, seed=935, workers=2)
    )
    rows = []
    for record in result.records:
        row = record.as_dict()
        row.pop("duration_ms", None)
        rows.append(json.dumps(row, sort_keys=True))
    assert hashlib.sha256("\n".join(rows).encode()).hexdigest() == _INERT_TOURNAMENTS[path]


# --------------------------------------------------------------------------- perft


def _oracle_perft(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    moves = oracle.legal(engine, state)
    if isinstance(moves, tuple):
        first, second = moves
        if depth == 1:
            return len(first) * len(second)
        return sum(
            _oracle_perft(engine, oracle.round_(engine, state, a, b), depth - 1)
            for a in first for b in second
        )
    if depth == 1:
        return len(moves)
    return sum(_oracle_perft(engine, oracle.turn(engine, state, m), depth - 1) for m in moves)


# Frozen after the oracle agreed (Q935a): from the opening, and from E1's position, which
# crosses the reply turn into announced rounds inside the tree.
_PERFT = {
    ("_spy_blast", "opening", 3): 2442,
    ("_spy_small", "e1", 3): 1427,
    ("_spy_small", "e1", 4): 31531,
}


def _e1(engine: Engine) -> GameState:
    return _position(engine, {"a1": "K", "a4": "R", "c2": "S"}, {"d5": "K", "c4": "S"})


@pytest.mark.parametrize("name, where, depth", sorted(_PERFT))
def test_perft_spy_agrees_with_the_oracle_and_is_frozen(name: str, where: str, depth: int) -> None:
    engine = _engine(name)
    state = engine.initial_state() if where == "opening" else _e1(engine)
    counted = engine.perft(state, depth)
    assert counted == _oracle_perft(engine, state, depth)
    assert counted == _PERFT[(name, where, depth)]


# ------------------------------------------------------------------- random players


@pytest.mark.parametrize("name", SPECS)
def test_random_plays_every_phase_the_same_in_the_core_and_in_python(name: str) -> None:
    from chess_variant_sandbox.tournament.play import play_game, play_game_python, replay

    engine = _engine(name)
    phases: set[str] = set()
    for seed in range(6):
        native = play_game(engine, ("random", "random"), seed)
        python = play_game_python(engine, ("random", "random"), seed)
        assert native.moves == python.moves
        assert (native.winner, native.reason, native.plies) == (python.winner, python.reason, python.plies)
        assert native.branching == python.branching and native.captures == python.captures
        assert native.material == python.material and native.nodes == python.nodes
        states = replay(engine, native)
        phases |= {engine.phase(state).name for state in states[:-1]}
    assert "alternating" in phases and (phases & {"announced", "blind"}), phases


def test_matrix_and_mcts_are_refused_on_a_spy_game() -> None:
    """Q935b lifted the refusal of greedy and minimax (`tests/test_spy_players.py`); the two
    names a spy game has no use for stay refused (`plan-spy.md` §3.3)."""
    from chess_variant_sandbox.tournament.play import play_game

    engine = _engine("_spy_small")
    for agent in ("matrix:nodes=100", "mcts:nodes=50"):
        with pytest.raises(ValueError, match="spy game"):
            play_game(engine, ("random", agent), 0)


def test_the_cheap_gate_plays_a_spy_game() -> None:
    playouts = _engine("_spy_stock").core.random_playouts(20, 0, 10.0, 1)
    assert len(playouts) == 20 and all(plies > 0 for plies, *_ in playouts)


# ------------------------------------------------------------------ the words and surfaces


def test_the_rules_say_the_phases_first() -> None:
    from chess_variant_sandbox.spec.describe import describe_rules, describe_spec

    lines = describe_rules(_load("_spy_resting"))
    phases = next(index for index, line in enumerate(lines) if line.startswith("The spy (S)"))
    assert "announces their move first" in lines[phases]
    assert "reply move" in lines[phases]
    assert not any(line.startswith("Both players move at once") for line in lines)
    assert any("it goes to the announcer" in line for line in lines)
    assert "is a spy" in describe_spec(_load("_spy_small"))


def test_a_spy_costs_rule_mass_and_distance() -> None:
    from chess_variant_sandbox.spec.complexity import measure_complexity
    from chess_variant_sandbox.spec.distance import features

    spy, rounds = _load("_spy_small"), _load("_spy_small")
    for piece in rounds["pieces"]:
        piece.pop("spy", None)
    assert measure_complexity(spy).rule_mass > measure_complexity(rounds).rule_mass
    assert features(spy)["spy_types"] == 1.0 and features(rounds)["spy_types"] == 0.0


def test_relate_tells_a_spy_game_from_its_round_twin() -> None:
    from chess_variant_sandbox.spec.relate import normal_form, rules_involved

    spy, rounds = _load("_spy_small"), _load("_spy_small")
    for piece in rounds["pieces"]:
        piece.pop("spy", None)
    assert "piece.spy" in rules_involved(normal_form(spy))
    assert "piece.spy" not in rules_involved(normal_form(rounds))


def test_fairy_stockfish_refuses_a_spy() -> None:
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    with pytest.raises(Untranslatable) as refused:
        translate(_load("_spy_small"))
    assert any(reason.startswith("pieces[].spy") for reason in refused.value.reasons)


def test_the_page_plays_a_spy_game() -> None:
    """Since Q935c (`plan-spy.md` §3.6; its tests are `tests/test_play_spy.py`); the loop's
    screen plays one too since Q935d (`tests/test_spy_measured.py`)."""
    from chess_variant_sandbox.web.library import Game
    from chess_variant_sandbox.web.play import Sessions

    game = Game(hash="spy", name="spy", description="", path=HERE, source="test",
                spec=_load("_spy_small"))
    session = Sessions().start(game, ("human", "human"))
    assert session.snapshot()["spy"] == {"phase": "alternating", "announcer": None}


def test_the_play_command_walks_the_phases(capsys: pytest.CaptureFixture[str]) -> None:
    from chess_variant_sandbox.cli import main

    assert main(["play", str(HERE / "_spy_small.json"), "--agents", "random", "random", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "[alternating]" in out and "(announced)" in out
