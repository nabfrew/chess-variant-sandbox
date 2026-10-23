"""The learned evaluator's goal terms, read by the core and by the Python oracle (F10, Q752).

`rust/src/features.rs` is the port and `agents/features.py` the oracle (CLAUDE.md rule 3).
A term is read only by a weighted player (`minimax:h=`), and the audit (`docs/plan-tests.md`
§2.3, finding T8) found four of the fifteen that no test ever evaluated in the core:
`territory_margin`, `extinction_margin`, `hand_material` and `promotion_occupancy`, with
`check_margin` and the region count of `score_margin` beside them. Each arm here is an inline
spec that has the rule the term reads, and asks three things of it:

* **The value, to the bit.** A player weighting that one term and nothing else, searched one
  ply deep (`max_depth=1`) at a budget it never exhausts, values every position of a game:
  `Core.search_values` against the oracle's own search (`MinimaxAgent._negamax`) from the
  same shuffled root. Every leaf of that search is the term alone, so a core that reads the
  term differently returns a different number, whatever the moves would have been.
* **A played game.** The same term beside material, played through the core and through the
  Python loop, move for move with node counts (`test_backends.py`'s comparison).
* **The guards**, without which the two above could pass on a term that reads zero: the term
  reads non-zero on the arm's positions, and weighting it changes the game played.

Each arm weights only its own term in the value comparison, so a mutant of one term fails
that term's arms and no other (`docs/plan-tests.md` §8 F10 records the mutants).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents.base import build_agent, native_spec
from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator
from chess_variant_sandbox.agents.heuristics import Heuristic, register_heuristic
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import (
    eval_values,
    play_game,
    play_game_python,
    seat_rng,
)

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# Positions valued per arm: enough that every arm's term moves both ways, few enough that the
# oracle's one-ply search over them stays well inside the duration guard.
POSITIONS = 24
SEED = 5


def _gardner(**rules: Any) -> dict[str, Any]:
    spec: dict[str, Any] = load_spec(RULESETS / "gardner.json")
    spec["rules"].update(rules)
    return spec


def _promoting_zone() -> dict[str, Any]:
    """Gardner whose far three ranks are a zone, promoting only into a piece already lost.

    Gardner's own rule, the last rank and promotion forced, reads zero at every position: a
    pawn arriving is a queen before anything reads the board. A choice is not enough either,
    since a player counting material always takes the queen. `only_lost` (Grand chess's rule)
    offers nothing while both queens stand, so the pawns and knights that walk in (a pawn's
    first step, a knight's first leap) stand there unpromoted, and a knight leaps out again:
    the term moves both ways from the first plies.
    """
    spec = _gardner()
    spec["rules"]["promotion"] = [
        {"piece": "P", "ranks": [2, 3, 4], "to": ["Q"], "mandatory": False, "only_lost": True},
        {"piece": "N", "ranks": [2, 3, 4], "to": ["Q"], "mandatory": False, "only_lost": True},
    ]
    return spec


def _own_kinds() -> dict[str, Any]:
    """Schema 27's `per_type: "own"`: the armies count different kinds (`test_backends.py`)."""

    def leap(vector: list[int], symmetry: str) -> list[dict[str, Any]]:
        return [{"kind": "leap", "vectors": [vector], "symmetry": symmetry, "mode": "both"}]

    return {
        "schema_version": 27, "name": "own kinds", "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "A", "value": 3, "moves": leap([1, 0], "rotate4")},
            {"id": "B", "value": 3, "moves": leap([1, 1], "rotate4")},
            {"id": "C", "value": 3, "moves": leap([1, 2], "dihedral8")},
            {"id": "E", "value": 3, "moves": leap([2, 0], "rotate4")},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "A", "player": 0, "squares": ["b1", "e1"]},
            {"piece": "B", "player": 0, "squares": ["c1", "d1"]},
            {"piece": "C", "player": 1, "squares": ["b6", "e6"]},
            {"piece": "E", "player": 1, "squares": ["c6", "d6"]},
        ]},
        "rules": {"win": "none", "stalemate": "draw", "max_plies": 120, "repetition_draw": 3,
                  "extinction": {"pieces": ["A", "B", "C", "E"], "per_type": "own"}},
    }


# (id, term, spec). Each term's denominators and branches are arms of their own: a win
# threshold and the whole board for territory, one shared kind and two (the minimum across
# kinds is what a sum would get wrong) and each side's own kinds for extinction, a hand fed by
# captures and one stocked from the opening.
ARMS: list[tuple[str, str, Any]] = [
    ("territory-to-a-threshold", "territory_margin", lambda: _gardner(
        territory={"mode": "permanent", "win_at": 8})),
    ("territory-over-the-board", "territory_margin", lambda: _gardner(
        territory={"mode": "repaint"})),
    ("extinction-of-one-kind", "extinction_margin", lambda: _gardner(
        extinction={"pieces": ["P"], "min_count": 1, "result": "loss"})),
    ("extinction-of-two-kinds", "extinction_margin", lambda: _gardner(
        extinction={"pieces": ["P", "N"], "min_count": 1, "result": "loss"})),
    ("extinction-of-own-kinds", "extinction_margin", _own_kinds),
    ("a-hand-from-captures", "hand_material", lambda: _gardner(
        drops={"from_captures": True})),
    ("a-hand-stocked", "hand_material", lambda: _gardner(
        drops={"from_captures": False, "stock": {"N": 1}})),
    ("a-promotion-zone", "promotion_occupancy", _promoting_zone),
    ("a-check-limit", "check_margin", lambda: _gardner(check_limit=3)),
    ("a-region-scored", "score_margin", lambda: _gardner(
        scoring={"count": "region", "region": ["c3", "b4", "c4", "d4"]}, max_plies=16)),
    ("a-region-scored-late", "score_urgency", lambda: _gardner(
        scoring={"count": "region", "region": ["c3", "b4", "c4", "d4"]}, max_plies=16)),
    ("territory-scored", "score_margin", lambda: _gardner(
        territory={"mode": "permanent"}, scoring={"count": "territory"})),
]
IDS = [arm[0] for arm in ARMS]

# Who the term's player meets and at which seed, greedy at `SEED` unless named. The own-kinds
# game reads its margin non-zero for at most a ply or two against greedy or a search (two of
# each kind a side: the first capture is answered and then the game ends or repeats); against
# random at seed 2 it reads non-zero at six of its thirteen positions.
OPPONENTS = {"extinction-of-own-kinds": ("random", 2)}

# The arms whose term moves no move at 150 nodes, each with the reason, so the guard on the
# played game below does not hold them. Own kinds: every kind is worth 3 and each side starts
# with two of each, so material and the margin rise and fall together and the search never
# separates them (seeds 1-5, weights 2-8, against greedy, random and minimax). Its term is
# held by value above, and `test_backends.py`'s own-kinds arm plays it against the built-in.
MOVES_NO_MOVE = {"extinction-of-own-kinds"}


def _vector(**terms: float) -> tuple[float, ...]:
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)


def _heuristic(weights: tuple[float, ...], name: str) -> str:
    heuristic = Heuristic(name=name, plan="", weights=weights, values=())
    register_heuristic(heuristic)
    return heuristic.digest


def _term_alone(term: str) -> str:
    """The term at unit weight and nothing else, one ply deep, at a budget it never meets."""
    return f"minimax:h={_heuristic(_vector(**{term: 1.0}), term)},nodes=1000000,max_depth=1"


def _beside_material(term: str) -> str:
    weights = _vector(material=1.0, mobility=0.05, **{term: 2.0})
    return f"minimax:h={_heuristic(weights, f'material-and-{term}')},nodes=150"


def _material_only() -> str:
    return f"minimax:h={_heuristic(_vector(material=1.0, mobility=0.05), 'material')},nodes=150"


def _agents(label: str, term: str) -> tuple[str, str]:
    return _beside_material(term), OPPONENTS.get(label, ("greedy", SEED))[0]


def _seed(label: str) -> int:
    return OPPONENTS.get(label, ("greedy", SEED))[1]


def _positions(engine: Engine, label: str, term: str) -> list[Any]:
    """The positions of a game the term's own player plays, up to `POSITIONS`."""
    record = play_game(engine, _agents(label, term), seed=_seed(label))
    state = engine.initial_state()
    states = []
    for move in record.moves[:POSITIONS]:
        states.append(state)
        state = engine.apply(state, move)
    return states


def _oracle_value(engine: Engine, player: str, state: Any, seed: int) -> tuple[float, int]:
    """The oracle's one-ply root value and move, as `Player::search_value` reaches them.

    The root is `search_root`'s (`agents.rs`, `MinimaxAgent._search_root`): the moves shuffled
    by the seat's stream for `seed`, each child searched by the oracle's own `_negamax` at
    depth 0 (its quiescence), the first strict maximum kept.
    """
    agent: Any = build_agent(player, engine)
    agent._begin_search()
    ordered = list(engine.legal_moves(state))
    seat_rng(seed, 0).shuffle(ordered)
    alpha, best = -math.inf, ordered[0]
    for move in ordered:
        child = engine.apply(state, move)
        if child.side_to_move != state.side_to_move:
            score = -agent._negamax(child, 0, 1, -math.inf, -alpha)
        else:
            score = agent._negamax(child, 0, 1, alpha, math.inf)
        if score > alpha:
            alpha, best = score, move
    return alpha, best


@pytest.mark.parametrize("label,term,build", ARMS, ids=IDS)
def test_the_term_alone_values_each_position_the_same_on_both_sides(
    label: str, term: str, build: Any
) -> None:
    engine = Engine(build())
    _assert_valued_alike(engine, _term_alone(term), _positions(engine, label, term))


# Games that declare none of the rules the goal terms read, valued by a player weighting all
# fifteen terms at once: each term's "no such rule here" branch is then read on both sides,
# where it must be exactly nothing, and a stray number in one of them moves the value. Plain
# gardner has no goal rule; own kinds has no royal and no promotion either.
INERT: list[tuple[str, Any]] = [("plain-gardner", _gardner), ("own-kinds", _own_kinds)]


@pytest.mark.parametrize("label,build", INERT, ids=[arm[0] for arm in INERT])
def test_every_term_at_once_values_a_game_without_their_rules_the_same_on_both_sides(
    label: str, build: Any
) -> None:
    engine = Engine(build())
    every = _heuristic((1.0,) * len(VOCABULARY), "every-term")
    record = play_game(engine, (_material_only(), "greedy"), seed=SEED)
    states, state = [], engine.initial_state()
    for move in record.moves[:POSITIONS]:
        states.append(state)
        state = engine.apply(state, move)
    _assert_valued_alike(engine, f"minimax:h={every},nodes=1000000,max_depth=1", states)


# F10b (Q752b): the terms' degenerate corners, on hand-built positions: an army with nothing
# but its king (no piece to advance, to spread, to share material against), a race whose
# goals only one side has (`mirror: false`), and a scored colouring with nothing coloured.
# Every term at once, as above, so each corner's zero is read on both sides.
DEGENERATE: list[tuple[str, Any, list[dict[str, tuple[str, int]]]]] = [
    ("bare-armies", _gardner, [
        {"a1": ("K", 0), "e5": ("K", 1)},
        {"a1": ("K", 0), "c2": ("P", 0), "e5": ("K", 1)},
        {"a1": ("K", 0), "c4": ("N", 1), "e5": ("K", 1)},
    ]),
    ("one-sided-race", lambda: _gardner(reach={"piece": "N", "squares": ["c5"],
                                               "mirror": False, "result": "win"}), [
        {"a1": ("K", 0), "b1": ("N", 0), "e5": ("K", 1), "d5": ("N", 1)},
        {"a1": ("K", 0), "e5": ("K", 1)},
    ]),
    ("scored-colouring", lambda: _gardner(territory={"mode": "permanent"},
                                          scoring={"count": "territory"}), [
        {"a1": ("K", 0), "b2": ("R", 0), "e5": ("K", 1)},
    ]),
]


@pytest.mark.parametrize("label,build,placements", DEGENERATE,
                         ids=[arm[0] for arm in DEGENERATE])
def test_every_term_at_once_values_a_degenerate_position_the_same_on_both_sides(
    label: str, build: Any, placements: list[dict[str, tuple[str, int]]]
) -> None:
    engine = Engine(build())
    every = _heuristic((1.0,) * len(VOCABULARY), "every-term")
    states = [engine.make_state(placement, side_to_move=side)
              for placement in placements for side in (0, 1)]
    _assert_valued_alike(engine, f"minimax:h={every},nodes=1000000,max_depth=1", states)


def _assert_valued_alike(engine: Engine, player: str, positions: list[Any]) -> None:
    """`Core.search_values` against `_oracle_value` at each position that is not over."""
    states = [state for state in positions if engine.terminal(state) is None]
    seeds = [SEED + index for index in range(len(states))]
    native = native_spec(player, engine.spec.hash)
    assert native is not None
    answers = engine.core.search_values(
        [_pack(state) for state in states], native, eval_values(engine.spec), seeds
    )
    for state, seed, answer in zip(states, seeds, answers, strict=True):
        assert answer is not None, "a budget of a million never finishes one ply?"
        value, move, depth = answer
        assert depth == 1
        assert (value, move) == _oracle_value(engine, player, state, seed), f"ply {state.ply}"


@pytest.mark.parametrize("label,term,build", ARMS, ids=IDS)
def test_the_term_reads_something_on_the_arms_positions(
    label: str, term: str, build: Any
) -> None:
    """The guard on the comparison above: a term that is zero everywhere compares nothing.

    The oracle reads the two players differently at some position, so the term is non-zero
    there, and a core that read it from the wrong side, or as a constant, would not agree.
    """
    engine = Engine(build())
    reader = FeatureEvaluator.for_engine(engine, (1.0,) * len(VOCABULARY))
    index = VOCABULARY.index(term)
    differ = [
        state.ply
        for state in _positions(engine, label, term)
        if reader.features(engine, state, 0)[index] != reader.features(engine, state, 1)[index]
    ]
    assert len(differ) >= 2, f"{term} reads both players alike at all but {len(differ)} plies"


@pytest.mark.parametrize("label,term,build", ARMS, ids=IDS)
def test_a_player_weighting_the_term_plays_the_same_game_on_both_sides(
    label: str, term: str, build: Any
) -> None:
    engine = Engine(build())
    agents = _agents(label, term)
    native = play_game(engine, agents, seed=_seed(label))
    in_python = play_game_python(engine, agents, seed=_seed(label))
    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum


@pytest.mark.parametrize(
    "label,term,build",
    [arm for arm in ARMS if arm[0] not in MOVES_NO_MOVE],
    ids=[label for label in IDS if label not in MOVES_NO_MOVE],
)
def test_weighting_the_term_changes_the_game(label: str, term: str, build: Any) -> None:
    """The guard on the played comparison: the term moves a move at 150 nodes."""
    engine = Engine(build())
    weighted, opponent = _agents(label, term)
    control = _material_only()
    seed = _seed(label)
    assert play_game(engine, (weighted, opponent), seed=seed).moves != play_game(
        engine, (control, opponent), seed=seed
    ).moves
