"""Schema 29's rest in the players, the analysis pass and the page (Q983b, `docs/plan-rest.md`
§5.4, §5.6, §7).

- **The term.** In rounds, `eval.rs::rested_royal_falls` and its oracle read the resting list:
  a side's only royal resting (an entry, the rest not lifted) where a standing enemy piece's
  legal move lands. With every type resting 1 at most one piece rests a player, so on the
  rewritten round specs the term is schema 27's one-square term exactly; with longer rests
  (breather by weight) the core and the oracle still agree, value for value. An alternating
  game has no term: its evaluator's mobility counts the core's lists, which the rest filters.
- **The plane.** The learned evaluator's `rest` plane is `2 x S x R` one-hot per (turns left,
  player, square), in every game that rests; for `R = 1` a round net's layout is schema 27's.
- **The tallies.** `analysis.REST_SERIES`, one entry a side-turn, the core's answer
  (`Engine.rest_turn`), held to `tests/rest_oracle.py`; absent from a game without a rest.
- **The page.** The resting pieces in every game, numbered where a rest is longer than one,
  on a step back too; a lifted turn said; the cards' rest.
"""

from __future__ import annotations

import json
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import rest_oracle
from test_matrix_players import core_round, oracle_round
from test_matrix_players import walk as round_walk
from test_web import _in_page

from chess_variant_sandbox.agents.evaluation import Evaluator, rested_royal_falls
from chess_variant_sandbox.agents.nets import TURN_CAP, Layout, NetEvaluator, random_net
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import analysis as analysis_pass
from chess_variant_sandbox.tournament.analysis import REST_SERIES, CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.play import play_game
from chess_variant_sandbox.web.library import Game, pieces
from chess_variant_sandbox.web.play import HUMAN, Session, Sessions

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# Every type rests 1 (the rewritten round games), and the longer and per-type rests.
BREATHER = "archive/breather.json"
BY_WEIGHT = "tests/_retired_breather_by_weight.json"  # rounds: N, B, K 1; R, Q 2
WEARY_HEAVIES = "tests/_retired_gardner_with_weary_heavies.json"  # turns: Q 2, R 1
WEARY_QUEEN = "archive/chess-with-a-weary-queen.json"  # turns: Q 1
CHECKMATE = "tests/_rest_checkmate.json"  # turns: Q 2, R 1, N 1
DROPS = "tests/_rest_drops.json"
# Q942a's four round test specs (`test_round_players_rest.CHASE_SPECS`).
CHASE_SPECS = ("resting", "collide", "atomic_collide", "scored")


def _engine(path: str) -> Engine:
    return Engine(load_spec(ROOT / path))


def _chase(name: str) -> Engine:
    return Engine(load_spec(HERE / f"_simultaneous_{name}.json"))


def _placed(
    engine: Engine,
    pieces_at: dict[str, tuple[str, int]],
    rest: tuple[tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]] = ((), ()),
) -> GameState:
    """A position with these pieces and these resting pieces (`(square name, turns left)`),
    keyed by the slow hash."""
    rested = tuple(
        tuple(sorted((engine._square_index(name), left) for name, left in side)) for side in rest
    )
    state = engine.make_state(pieces_at).with_(rest=rested)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def _alternating_walk(engine: Engine, seed: int, plies: int = 80) -> tuple[list[int], list[GameState]]:
    rng = random.Random(seed)
    state = engine.initial_state()
    moves, states = [], [state]
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        move = rng.choice(engine.legal_moves(state))
        state = engine.apply(state, move)
        moves.append(move)
        states.append(state)
    return moves, states


# ------------------------------------------------------------------ the players' term


@pytest.mark.parametrize("name", CHASE_SPECS)
def test_with_every_rest_at_one_the_term_reads_schema_27s_one_square(name: str) -> None:
    """On the rewritten round specs every type rests 1, so a player has at most one resting
    piece and the term over the list is the old term over one rested square, at every
    position a game passes through."""
    engine = _chase(name)
    assert set(engine.spec.rest) <= {0, 1}
    for seed in (3, 4, 5):
        for state in round_walk(engine, 40, seed=seed):
            lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
            for side in (0, 1):
                assert len(state.rest[side]) <= 1
                rested = state.rest[side][0][0] if state.rest[side] else -1
                one_square = rested >= 0 and rested_royal_falls(
                    engine.spec, state.with_(rest=tuple(
                        ((rested, 1),) if p == side else () for p in (0, 1)
                    )), side, lists)
                assert rested_royal_falls(engine.spec, state, side, lists) == one_square


def test_a_royal_resting_two_turns_beside_another_resting_piece_falls() -> None:
    """Breather by weight (rounds, rests up to 2): the king resting with one round still to
    go, its rook resting too, is reached by a standing bishop; lifted, it is not."""
    engine = _engine(BY_WEIGHT)
    at = {"e1": ("K", 0), "d2": ("B", 0), "a1": ("R", 0),
          "b4": ("K", 1), "e4": ("P", 1), "a5": ("R", 1)}
    state = _placed(engine, at, rest=((), (("a5", 2), ("b4", 1))))
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    assert rested_royal_falls(engine.spec, state, 1, lists)
    assert not rested_royal_falls(engine.spec, state, 0, lists)
    later = _placed(engine, at, rest=((), (("a5", 1), ("b4", 2))))
    assert rested_royal_falls(engine.spec, later, 1, (lists[0], engine.legal_moves_for(later, 1)))
    # The rest lifted (nothing else of player 1's can move): the king may run, so no fall.
    alone = {"e1": ("K", 0), "d2": ("B", 0), "b4": ("K", 1)}
    lifted = _placed(engine, alone, rest=((), (("b4", 2),)))
    assert engine.rest_turn(lifted, 1).lifted
    lists = (engine.legal_moves_for(lifted, 0), engine.legal_moves_for(lifted, 1))
    assert not rested_royal_falls(engine.spec, lifted, 1, lists)


def test_the_core_and_the_oracle_agree_where_rests_run_longer() -> None:
    """Every value the round players compute, core against oracle, over positions where
    pieces of breather by weight rest one and two rounds."""
    engine = _engine(BY_WEIGHT)
    states = round_walk(engine, 6, seed=21)
    # And a royal resting beside a second piece resting two rounds, which the term prices.
    states.append(_placed(engine, {"e1": ("K", 0), "d2": ("B", 0), "a1": ("R", 0),
                                   "b4": ("K", 1), "e4": ("P", 1), "a5": ("R", 1)},
                          rest=((), (("a5", 2), ("b4", 1)))))
    for index, state in enumerate(states):
        for agent in ("greedy", "matrix:nodes=200"):
            for seat in (0, 1):
                core = core_round(engine, agent, state, seat, seed=index)
                assert core == oracle_round(engine, agent, state, seat, seed=index), (
                    agent, index, seat,
                )


def test_an_alternating_evaluators_mobility_counts_the_cores_filtered_list() -> None:
    """No term in alternating play (§5.4): a search sees the next turn's moves, which the core
    filters. So the evaluator's mobility must count those moves, not the board's: a resting
    queen's moves are not the side to move's."""
    engine = _engine(CHECKMATE)
    at = {"a1": ("K", 0), "c3": ("Q", 0), "c1": ("N", 0), "e5": ("K", 1), "a5": ("P", 1)}
    state = _placed(engine, at, rest=((("c3", 2),), ()))
    held = engine.legal_moves(state)
    free = engine.legal_moves(state.with_(rest=((), ())))
    assert len(held) < len(free) and engine.rest_turn(state, 0).binds
    evaluator = Evaluator.for_engine(engine)
    assert evaluator.mobility_weight
    assert evaluator.evaluate(engine, state, 0) == evaluator.evaluate(engine, state, 0, len(held))
    assert evaluator.evaluate(engine, state, 0) != evaluator.evaluate(engine, state, 0, len(free))


# ------------------------------------------------------------------ the core's answer


def _oracle_turn(engine: Engine, state: GameState, seat: int | None) -> tuple[bool, bool]:
    held = sorted(rest_oracle.legal(engine, state, seat))
    free = sorted(rest_oracle.legal(engine, state.with_(rest=((), ())), seat))
    binds = held != free
    return binds, (not binds) and rest_oracle.lifted(engine, state, seat)


@pytest.mark.parametrize("path", [CHECKMATE, DROPS, WEARY_HEAVIES, WEARY_QUEEN])
def test_rest_turn_is_the_oracles_on_every_turn_of_a_game(path: str) -> None:
    engine = _engine(path)
    seen = {"binds": 0, "lifted": 0}
    for seed in (1, 2):
        _, states = _alternating_walk(engine, seed, plies=120)
        for state in states:
            if engine.terminal(state) is not None:
                continue
            turn = engine.rest_turn(state, state.side_to_move)
            assert (turn.binds, turn.lifted) == _oracle_turn(engine, state, None)
            seen["binds"] += turn.binds
            seen["lifted"] += turn.lifted
    assert seen["binds"] > 0


@pytest.mark.parametrize("path", [BREATHER, BY_WEIGHT])
def test_rest_turn_is_the_oracles_for_each_seat_of_a_round(path: str) -> None:
    engine = _engine(path)
    bound = 0
    for state in round_walk(engine, 30, seed=8):
        for seat in (0, 1):
            turn = engine.rest_turn(state, seat)
            assert (turn.binds, turn.lifted) == _oracle_turn(engine, state, seat)
            bound += turn.binds
    assert bound > 0


def test_the_lift_is_reported_and_never_beside_binds() -> None:
    engine = _engine(CHECKMATE)
    # White's king boxed in by two rooks; its queen, resting, is all it can move.
    at = {"a1": ("K", 0), "c3": ("Q", 0), "b5": ("R", 1), "e2": ("R", 1), "e5": ("K", 1)}
    state = _placed(engine, at, rest=((("c3", 1),), ()))
    assert engine.rest_turn(state, 0).lifted and not engine.rest_turn(state, 0).binds
    assert rest_oracle.lifted(engine, state)
    assert any(move_from(m) == engine._square_index("c3") for m in engine.legal_moves(state))
    # Nothing rests in chess: both false, whatever the board.
    chess = _engine("rulesets/chess.json")
    turn = chess.rest_turn(chess.initial_state(), 0)
    assert not turn.binds and not turn.lifted


# ------------------------------------------------------------------ the plane


def _schema_27_round_layout(layout: Layout) -> tuple[int, int, int]:
    """Where schema 27's round segments sat: a 2 x S rest plane after the initiative."""
    s = layout.n_squares
    rest = layout.turn + TURN_CAP + (2 if layout.has_initiative else 0)
    void = rest + 2 * s
    return rest, void, void + 4 * s


@pytest.mark.parametrize("path", [BREATHER, "tests/_simultaneous_resting.json"])
def test_a_round_nets_layout_for_a_rest_of_one_is_unchanged(path: str) -> None:
    engine = _engine(path)
    layout = Layout.of(engine.spec)
    assert layout.rest_turns == 1
    assert (layout.rest, layout.void, layout.n_inputs) == _schema_27_round_layout(layout)
    assert engine.core.net_inputs() == layout.n_inputs
    s = layout.n_squares
    probe = engine.initial_state().with_(rest=(((1, 1),), ((s - 1, 1),)))
    marks = [f for f in layout.features(probe) if layout.rest <= f < layout.void]
    assert marks == [layout.rest + 1, layout.rest + s + s - 1]


@pytest.mark.parametrize(
    ("path", "turns"), [(WEARY_HEAVIES, 2), (WEARY_QUEEN, 1), (CHECKMATE, 2), (DROPS, None)]
)
def test_an_alternating_game_that_rests_has_the_plane_after_turn(path: str, turns: int | None) -> None:
    engine = _engine(path)
    layout = Layout.of(engine.spec)
    r = max(engine.spec.rest)
    assert turns is None or r == turns
    assert not layout.rounds and layout.has_rest and layout.rest_turns == r
    assert layout.rest == layout.turn + TURN_CAP
    assert layout.n_inputs == layout.rest + 2 * layout.n_squares * r
    assert engine.core.net_inputs() == layout.n_inputs


def _probes(engine: Engine) -> list[GameState]:
    """Every (turns left, player) block of the plane: on the opening board, each player's
    first pieces resting, as many as the longest rest allows (the core keeps no more, and none
    on an empty square)."""
    r = max(engine.spec.rest)
    start = engine.initial_state()
    spec = engine.spec
    own = [
        [sq for sq, value in enumerate(start.board) if value and spec.owner_of(value) == player]
        for player in (0, 1)
    ]
    out = []
    for left in range(1, r + 1):
        rest = tuple(
            tuple(sorted({own[player][0]: left, **({own[player][1]: r} if r > 1 else {})}.items()))
            for player in (0, 1)
        )
        out.append(start.with_(rest=rest))
    return out


@pytest.mark.parametrize("path", [WEARY_HEAVIES, CHECKMATE, DROPS])
def test_an_alternating_rest_line_is_the_oracles_to_the_bit(path: str) -> None:
    engine = _engine(path)
    layout = Layout.of(engine.spec)
    moves, states = _alternating_walk(engine, seed=6, plies=60)
    assert any(state.rest[0] or state.rest[1] for state in states)
    blob = random_net(layout.n_inputs, seed=3)
    oracle = NetEvaluator(engine, blob, ())
    answers = engine.core.net_line(_pack(engine.initial_state()), blob, moves)
    assert len(answers) == len(states)
    for index, ((raw, inputs, fresh), state) in enumerate(zip(answers, states, strict=True)):
        assert inputs == layout.features(state), index
        assert raw == fresh == oracle.raw(state), index
    hit = set()
    for probe in _probes(engine):
        ((raw, inputs, fresh),) = engine.core.net_line(_pack(probe), blob, [])
        assert inputs == layout.features(probe)
        assert raw == fresh == oracle.raw(probe)
        hit |= {(f - layout.rest) // layout.n_squares for f in inputs if f >= layout.rest}
    assert hit == set(range(2 * layout.rest_turns))


def test_a_round_line_where_rests_run_longer_is_the_oracles_to_the_bit() -> None:
    engine = _engine(BY_WEIGHT)
    layout = Layout.of(engine.spec)
    assert layout.rounds and layout.rest_turns == 2
    assert layout.void == layout.rest + 4 * layout.n_squares
    states = round_walk(engine, 10, seed=4)
    blob = random_net(layout.n_inputs, seed=5)
    oracle = NetEvaluator(engine, blob, ())
    for state in states + _probes(engine):
        ((raw, inputs, fresh),) = engine.core.net_line(_pack(state), blob, [])
        assert inputs == layout.features(state)
        assert raw == fresh == oracle.raw(state)


# ------------------------------------------------------------------ the tallies

# The fields an analysis held before schema 29's tallies, which a game without a rest still
# writes, and only these.
_BEFORE = ["seed", "agents", "near_best", "legal_counts", "moves_by_type", "captures_by_type",
           "capture_counts", "revisited_plies", "walked_plies", "starting_instances",
           "moved_instances", "position_keys"]


def _analysed(engine: Engine, seed: int = 1) -> tuple[Any, GameAnalysis]:
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    record = play_game(engine, ("greedy", "random"), seed)
    found = analysis_pass._analyse_game(engine, evaluator, record,
                                        analysis_pass._epsilon_for(engine))
    return record, found


def test_a_game_without_a_rest_writes_the_bytes_it_did(tmp_path: Path, monkeypatch) -> None:
    engine = _engine("rulesets/gardner.json")

    def refused(*_args: Any) -> Any:
        raise AssertionError("a game without a rest asks the core nothing about one")

    monkeypatch.setattr(Engine, "rest_turn", refused)
    _, found = _analysed(engine)
    assert found.rest is None
    assert list(found.as_dict()) == _BEFORE
    before = {f.name: getattr(found, f.name) for f in fields(found) if f.name in _BEFORE}
    before["agents"] = list(found.agents)
    path = CorpusAnalysis(games=[found]).save(tmp_path / "analysis.json")
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"games": [before], "blunders": []}, indent=2)


@pytest.mark.parametrize("path", [CHECKMATE, WEARY_HEAVIES])
def test_each_turns_tally_is_the_cores_answer_and_the_oracles(path: str) -> None:
    engine = _engine(path)
    record, found = _analysed(engine, seed=2)
    tally = found.rest
    assert tally is not None and list(tally) == list(REST_SERIES)
    assert all(len(series) == record.plies for series in tally.values())
    state = engine.initial_state()
    for ply, move in enumerate(record.moves):
        seat = state.side_to_move
        binds, lifted = _oracle_turn(engine, state, None)
        row = {key: tally[key][ply] for key in REST_SERIES}
        assert row == {"side": seat, "binds": int(binds), "lifted": int(lifted),
                       "resting": len(state.rest[seat])}, ply
        state = engine.apply(state, move)
    assert sum(tally["binds"]) > 0 and sum(tally["resting"]) > 0
    assert not any(b and lift for b, lift in zip(tally["binds"], tally["lifted"], strict=True))
    assert GameAnalysis.from_dict(json.loads(json.dumps(found.as_dict()))) == found


def test_a_round_counts_each_seat() -> None:
    engine = _engine(BY_WEIGHT)
    record, found = _analysed(engine, seed=3)
    tally = found.rest
    assert tally is not None
    assert tally["side"] == [0, 1] * (len(record.moves) // 2)
    state = engine.initial_state()
    for index in range(0, len(record.moves) - 1, 2):
        for seat in (0, 1):
            binds, lifted = _oracle_turn(engine, state, seat)
            assert tally["binds"][index + seat] == int(binds)
            assert tally["lifted"][index + seat] == int(lifted)
            assert tally["resting"][index + seat] == len(state.rest[seat])
        state = engine.apply_round(state, record.moves[index], record.moves[index + 1])
    assert sum(tally["binds"]) > 0


def test_a_spy_game_that_rests_counts_its_turns_and_rounds() -> None:
    engine = _engine("tests/_retired_gardner_with_spies_resting.json")
    record, found = _analysed(engine, seed=4)
    tally = found.rest
    assert tally is not None
    assert len(tally["side"]) == len(record.moves)
    assert sum(tally["binds"]) > 0


# ------------------------------------------------------------------ the page


def _game(path: str) -> Game:
    spec = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return Game(hash=Path(path).stem, name=spec["name"], description="", path=ROOT / path,
                source="test", spec=spec)


def _session(path: str) -> Session:
    return Sessions().start(_game(path), (HUMAN, HUMAN), seed=0, scratch=True)


def _move(engine: Engine, state: GameState, frm: str, to: str, seat: int | None = None) -> int:
    moves = engine.legal_moves(state) if seat is None else engine.legal_moves_for(state, seat)
    found = [m for m in moves if move_from(m) == engine._square_index(frm)
             and move_to(m) == engine._square_index(to)]
    assert found, (frm, to)
    return found[0]


class TestThePage:
    def test_an_alternating_game_badges_its_resting_pieces_with_their_turns_left(self) -> None:
        session = _session(WEARY_HEAVIES)
        engine = session.engine
        opening = session.snapshot()
        assert opening["ruleset"]["rest"] == list(engine.spec.rest)
        assert opening["resting"] == []
        session.play(_move(engine, session.state, "d2", "d3"))
        snapshot = session.play(_move(engine, session.state, "e4", "d3"))  # black's pawn takes
        # A pawn rests nothing (rest 0). Now the queen takes back, and rests two turns.
        assert snapshot["resting"] == []
        snapshot = session.play(_move(engine, session.state, "d1", "d3"))
        queen = engine._square_index("d3")
        assert snapshot["resting"] == [{"player": 0, "square": queen, "left": 2}]
        assert snapshot["history"][-1]["resting"] == snapshot["resting"]
        assert "lifted" not in snapshot["history"][-1]
        assert all(m["from"] != queen for m in snapshot["moves"]) or snapshot["sideToMove"] == 1
        drawn = _in_page("FRAME = flatView(5, 5);", f"restsHtml({json.dumps(snapshot['resting'])}, 10, "
                             f"restNumbered({json.dumps(snapshot['ruleset']['rest'])}))")
        assert 'data-mark="rested"' in drawn and 'data-left="2"' in drawn and ">2</text>" in drawn

    def test_a_game_whose_rests_are_all_one_draws_the_plain_sign(self) -> None:
        assert _in_page("", "restNumbered([1, 1, 0, 1])") is False
        assert _in_page("", "restNumbered([0, 2, 1])") is True
        assert _in_page("", "restNumbered([])") is False
        plain = _in_page("", "restBadge(0, 0, 10, 1)")
        assert "data-left" not in plain and plain.count("<rect") == 2
        numbered = _in_page("", "restBadge(0, 0, 10, 1, 3)")
        assert 'data-left="3"' in numbered and "<rect" not in numbered

    def test_the_legend_says_rest_in_any_game_not_last_round(self) -> None:
        legend = _in_page("", "legendEntries(new Set(['rested']), null, '', null)")
        (entry,) = legend
        assert "pause sign" in entry and "rests" in entry and "round" not in entry

    def test_a_step_back_badges_its_own_position(self) -> None:
        snapshot = {"history": [{"text": "Qd2", "resting": [{"player": 0, "square": 8,
                                                              "left": 2}]}],
                    "sideToMove": 1, "colour": [], "hands": []}
        beat = _in_page(f"const S = {json.dumps(snapshot)};",
                        "beatEntry('move', [0], null, S, 1, null, null).resting")
        assert beat == snapshot["history"][0]["resting"]
        start = _in_page(f"const S = {json.dumps(snapshot)};",
                         "beatEntry('move', [0], null, S, 0, null, null).resting")
        assert start == []

    def test_a_lifted_turn_is_said_beside_its_move(self) -> None:
        engine = _engine(CHECKMATE)
        session = Sessions().start(_game(CHECKMATE), (HUMAN, HUMAN), seed=0, scratch=True)
        at = {"a1": ("K", 0), "c3": ("Q", 0), "b5": ("R", 1), "e2": ("R", 1), "e5": ("K", 1)}
        session.state = _placed(session.engine, at, rest=((("c3", 1),), ()))
        snapshot = session.play(_move(engine, session.state, "c3", "c4"))
        entry = snapshot["history"][-1]
        assert entry["lifted"] is True
        assert entry["resting"] == [{"player": 0, "square": engine._square_index("c4"),
                                     "left": 2}]
        said = _in_page(f"const M = {json.dumps(entry)};", "liftedHtml(M)")
        assert "data-lifted" in said and "rest lifted" in said
        assert _in_page("", "liftedHtml({text: 'Qc4'})") == ""

    def test_a_lifted_round_is_said_in_the_rounds_sentences(self) -> None:
        session = _session(BREATHER)
        engine = session.engine
        at = {"a1": ("K", 0), "e5": ("K", 1), "e4": ("P", 1)}
        session.state = _placed(engine, at, rest=((("a1", 1),), ()))
        session.play(_move(engine, session.state, "a1", "a2", seat=0))
        snapshot = session.play(_move(engine, session.state, "e5", "d5", seat=1))
        first, second = snapshot["history"][-2:]
        assert first.get("lifted") is True and "lifted" not in second
        words = second["reveal"]["words"]
        side = snapshot["ruleset"]["sides"][0]
        assert any(f"every move {side} had was a resting piece's" in line.lower()
                   for line in words), words
        assert first["resting"] == second["resting"] == snapshot["resting"]

    def test_a_game_without_a_rest_sends_nothing_new(self) -> None:
        session = _session("rulesets/gardner.json")
        engine = session.engine
        snapshot = session.play(engine.legal_moves(session.state)[0])
        assert snapshot["ruleset"]["rest"] == [] and snapshot["resting"] == []
        assert "resting" not in snapshot["history"][-1]
        assert "lifted" not in snapshot["history"][-1]

    def test_a_cards_rest_is_its_pill_and_its_line(self) -> None:
        cards = pieces(_game(WEARY_HEAVIES))
        by_id = {card["id"]: card for card in cards}
        assert by_id["Q"]["rest"] == 2 and by_id["R"]["rest"] == 1 and by_id["P"]["rest"] == 0
        pill = _in_page(f"const P = {json.dumps(by_id['Q'])};", "restPillHtml(P)")
        assert 'data-rest="2"' in pill and "rests 2 turns after it moves" in pill
        assert _in_page(f"const P = {json.dumps(by_id['P'])};", "restPillHtml(P)") == ""
        assert "rest" not in pieces(_game("rulesets/gardner.json"))[0]
