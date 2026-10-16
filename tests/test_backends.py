"""What holds the core honest now that there is only one engine.

There used to be a second implementation -- `engine/reference.py`, a pure-Python
interpreter of the same compiled tables -- and this file compared the two. That test was
doing two jobs at once and they came apart: it checked that the *movegens* agreed and
that the *agents* agreed, which meant a compensating pair of bugs in both could cancel
and still pass. The interpreter is gone; the two jobs are separate here.

**The agents** are compared across the boundary, not across two engines. The Python game
loop with the Python agents, driven over the Rust engine, must produce the same game as
`core.play_game` from the same seed. Same ruleset, same movegen, same streams; the only
difference is which side of the boundary chooses the move. That isolates the search,
which is what the old test was really being trusted for -- every calibration number in
CLAUDE.md was measured with the Python agents, and a game is contracted to be a pure
function of `(ruleset_hash, seed, agent_configs)`.

**The movegen** is pinned by the perft literals in `test_variants.py` and
`test_schema_two.py` -- Fairy-Stockfish's numbers for the committed rulesets, a frozen
ladder for the inline probes -- and by the properties below, which a second interpreter
was never needed for: legal moves are a subset of pseudo-legal ones, the incremental
hash equals the slow definition, applying a move does not mutate the position it was
applied to, and a terminal verdict does not depend on whether the caller passed a move
list.

What that loses, deliberately, is the ply-by-ply comparison of two live generators: a
perft literal checks a count at a handful of depths from the opening, where that checked
the whole move *list*, in order, in every position a game visits. The properties below
are the same category of check without a second implementation to keep in step.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, RACE_SHARE, Evaluator
from chess_variant_sandbox.agents.features import VOCABULARY, builtin_vector, feature_context
from chess_variant_sandbox.agents.heuristics import Heuristic, register_heuristic
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import play_game, play_game_python

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# One anchor plus the rulesets that reach corners chess does not: a mid-board promotion
# rank, swapped pawn moves and en passant, a royal that is not a king-mover, a 5x5
# board, `stalemate: loss`. `bastion` is here for its cannon -- a hop attack turns on
# the branch of the legality filter that no other committed ruleset reaches, because
# adding a screen to a line can *create* a check rather than block one.
CORPUS = ["chess", "makruk", "berolina", "knightmate", "gardner", "shatranj", "bastion"]

# Budget is part of agent identity, so these are the players being compared, not
# stand-ins for them. The split is by what the *Python* side costs: it is the slow
# implementation now.
FAST_ROSTER = [
    ("random", "random"),
    ("greedy", "random"),
    ("minimax:nodes=150", "greedy"),
]
SLOW_ROSTER = [
    ("greedy", "greedy"),
    ("minimax:nodes=250", "minimax:nodes=250"),
]

# MCTS gets its own, narrower list. A rollout is 60 random plies and there are 30 of
# them per move, so a single game through the Python side is half a minute on an 8x8
# board -- enough to dominate the whole suite if it ran on all six. Two boards is
# already the interesting comparison: the smallest one anyone plays, and one whose
# mid-board promotion rank makes the rollouts hit rules chess never reaches.
MCTS_CORPUS = ["gardner", "makruk"]


def _engine(name: str) -> Engine:
    return Engine(load_spec(RULESETS / f"{name}.json"))


def _vector(**terms: float) -> tuple[float, ...]:
    """A weight vector named by its terms, at whatever length the vocabulary now is.

    Three vectors in this file used to be written out positionally, and adding a term to
    ``VOCABULARY`` broke five arms that had nothing to do with it. Worse than the
    breakage: a positional list that is one entry short is a list whose *last* terms have
    all shifted, so the arm goes on running and stops testing what it says.
    """
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)


# --------------------------------------------------------------------- the agents


def _assert_same_game(name: str, agents: tuple[str, str]) -> None:
    """The whole record, not just the result.

    ``nodes`` is in here deliberately: two searches can pick the same moves for
    different reasons, and a divergence in how many positions were visited means the
    budget -- which is what the strategic-depth metric compares -- means something
    different on the two sides. ``depth_sum`` and ``searches`` are there for the same
    reason one rung down: they say what those nodes *bought*, and two searches that
    agree on every move while disagreeing about how deep they got are two searches
    whose reported depth describes only one of them.
    """
    _assert_same_played(_engine(name), agents)


def _assert_same_played(engine: Engine, agents: tuple[str, str]) -> None:
    native = play_game(engine, agents, seed=5)
    in_python = play_game_python(engine, agents, seed=5)

    assert native.moves == in_python.moves
    assert (native.winner, native.reason) == (in_python.winner, in_python.reason)
    assert native.branching == in_python.branching
    assert native.material == in_python.material
    assert native.captures == in_python.captures
    assert native.nodes == in_python.nodes
    assert native.depth_sum == in_python.depth_sum
    assert native.searches == in_python.searches


@pytest.mark.parametrize("name", CORPUS)
@pytest.mark.parametrize("agents", FAST_ROSTER, ids=lambda pair: f"{pair[0]}-vs-{pair[1]}")
def test_a_played_game_is_identical_move_for_move(name: str, agents: tuple[str, str]) -> None:
    _assert_same_game(name, agents)


@pytest.mark.slow
@pytest.mark.parametrize("name", CORPUS)
@pytest.mark.parametrize("agents", SLOW_ROSTER, ids=lambda pair: f"{pair[0]}-vs-{pair[1]}")
def test_the_deeper_agents_also_agree_move_for_move(
    name: str, agents: tuple[str, str]
) -> None:
    _assert_same_game(name, agents)


@pytest.mark.slow
@pytest.mark.parametrize("name", MCTS_CORPUS)
def test_the_tree_search_agrees_move_for_move(name: str) -> None:
    """MCTS is the agent most likely to diverge and hardest to notice diverging.

    It draws from the RNG twice per expansion and once per rollout ply, it breaks ties
    on visit counts, and its UCT term banks wins from the child's mover's perspective.
    Any of those going subtly wrong still produces a legal, plausible game.
    """
    _assert_same_game(name, ("mcts:nodes=30", "random"))


# Schema 26 (Q918b): the round players. Q918a's test specs (`tests/_simultaneous_*.json`):
# a mirrored Gardner-sized game, drops and promotion, an explosion, an own-merge contact, a
# static portal pair beside a fold; and Q942a's schema-27 switches (the rest; all three with
# a score; swaps that collide beside an explosion). The fast arms cut each game at 24 plies (twelve rounds);
# the slow ones play it to its verdict.
ROUND_SPECS = ["gardner", "shogi", "atomic", "merge", "portals", "resting", "scored", "atomic_collide"]
ROUND_ROSTER = [
    ("greedy", "random"),
    ("matrix:nodes=250", "greedy"),
]


def _rounds(name: str, max_plies: int | None = None) -> Engine:
    spec = load_spec(Path(__file__).parent / f"_simultaneous_{name}.json")
    if max_plies is not None:
        spec["rules"]["max_plies"] = max_plies
    return Engine(spec)


@pytest.mark.parametrize("name", ROUND_SPECS)
@pytest.mark.parametrize("agents", ROUND_ROSTER, ids=lambda pair: f"{pair[0]}-vs-{pair[1]}")
def test_a_round_game_is_identical_move_for_move(name: str, agents: tuple[str, str]) -> None:
    _assert_same_played(_rounds(name, max_plies=24), agents)


@pytest.mark.slow
@pytest.mark.parametrize("name", ROUND_SPECS)
@pytest.mark.parametrize(
    "agents", [("greedy", "greedy"), ("matrix:nodes=800", "greedy")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_a_whole_round_game_is_identical_move_for_move(name: str, agents: tuple[str, str]) -> None:
    _assert_same_played(_rounds(name), agents)


def _turned(name: str, **turn: Any) -> Engine:
    """The same ruleset with a longer turn, so the turn is the only thing that changed."""
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = 4
    spec["rules"]["turn"] = turn
    return Engine(spec)


@pytest.mark.parametrize(
    "turn", [{"moves": 2}, {"moves": 2, "ramp": [1]}], ids=["two-moves", "ramped"]
)
@pytest.mark.parametrize(
    "agents",
    [("minimax:nodes=150", "greedy"), ("minimax:nodes=150", "minimax:nodes=150")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_a_side_that_moves_twice_is_searched_the_same_way_on_both_sides(
    turn: dict[str, Any], agents: tuple[str, str]
) -> None:
    """The check the negamax fix actually needs.

    Both implementations negate the child's value and flip the alpha-beta window at
    every ply, because negamax assumes the side alternates. A turn longer than one move
    makes that false, and the two sides fixing it *differently* -- or one of them not at
    all -- produces two legal, plausible, different games. Nothing else in the suite
    would see it: no committed ruleset has a longer turn, and a perft counts moves
    rather than choices.

    The ramped arm is the one that matters most. With a flat ``moves: 2`` the side
    changes on every second ply and a wrong-but-consistent rule could still line up; the
    ramp makes the answer differ between one ply and the next of the same game, which is
    why both sides ask "did the side change" rather than "is there a turn rule".
    """
    _assert_same_played(_turned("gardner", **turn), agents)


def _promoting_on_leaving(flag: bool) -> Engine:
    """Shogi with schema 15's ``on_leaving`` on every promotion entry, or without it.

    The committed `shogi.json` carries it on every entry since Q454 (B3). The flag is still
    written here either way, so that the two arms differ in it alone.
    """
    spec: dict[str, Any] = load_spec(RULESETS / "shogi.json")
    spec["schema_version"] = 15
    for entry in spec["rules"]["promotion"]:
        entry["on_leaving"] = flag
    return Engine(spec)


@pytest.mark.parametrize(
    "agents", [("random", "random"), ("greedy", "random")], ids=lambda pair: "-vs-".join(pair)
)
def test_a_promotion_out_of_the_zone_is_played_the_same_way_on_both_sides(
    agents: tuple[str, str],
) -> None:
    """The moves the flag adds are `MOVE_PROMOTION` words the Python agents must read as
    the core's do: loud, a material change, the piece it becomes."""
    _assert_same_played(_promoting_on_leaving(True), agents)


def test_the_on_leaving_arm_changes_the_game() -> None:
    """Otherwise the arm above would compare a game the flag never touched."""
    flagged = play_game(_promoting_on_leaving(True), ("random", "random"), seed=5)
    plain = play_game(_promoting_on_leaving(False), ("random", "random"), seed=5)
    assert flagged.moves != plain.moves


@pytest.mark.parametrize("name", ["rifle", "circe", "andernach", "kamikaze"])
@pytest.mark.parametrize(
    "agents", [("random", "random"), ("greedy", "random")], ids=lambda pair: "-vs-".join(pair)
)
def test_what_a_capture_leaves_behind_is_played_the_same_way_on_both_sides(
    name: str, agents: tuple[str, str]
) -> None:
    """Schema 17 (Q540): no library game leaves its capturer still, reborns a victim,
    switches a capturer's side or blasts nothing beside a capture. The Python agents must
    read each such capture as the core does: loud, a material change, the same choice."""
    engine = Engine(load_spec(RULESETS.parent / "archive" / f"{name}.json"))
    _assert_same_played(engine, agents)
    assert play_game(engine, agents, seed=5).moves != play_game(
        Engine(load_spec(RULESETS / "chess.json")), agents, seed=5
    ).moves


def _hexagon(name: str) -> Engine:
    """Schema 16's native hexagon (Q281a4): the twin, or its flanks joined by a seam (retired
    by Q870 for `pinwheel`, and rebuilt from the twin)."""
    from retired_specs import REBUILT

    if name in REBUILT:
        return Engine(REBUILT[name]())
    return Engine(load_spec(RULESETS.parent / "archive" / f"{name}.json"))


@pytest.mark.parametrize(
    "agents", [("random", "random"), ("greedy", "random")], ids=lambda pair: "-vs-".join(pair)
)
def test_a_move_across_a_hexagons_seam_is_played_the_same_way_on_both_sides(
    agents: tuple[str, str],
) -> None:
    """No committed ruleset has a hexagon's seam, and a crossing move is an ordinary
    `MOVE_NORMAL` word whose target the tables put on the far side: the Python agents must
    read it, and its capture, as the core does."""
    _assert_same_played(_hexagon("hex-glinski-flanks"), agents)


def test_the_hexagon_seam_arm_changes_the_game() -> None:
    """Otherwise the arm above would compare a game no crossing ever touched."""
    seamed = play_game(_hexagon("hex-glinski-flanks"), ("random", "random"), seed=5)
    plain = play_game(_hexagon("hex-glinski-hexagon"), ("random", "random"), seed=5)
    assert seamed.moves != plain.moves


def _with_rules(name: str, version: int, **rules: Any) -> Engine:
    """The same ruleset with extra rules bolted on, so the rule is the only difference."""
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = version
    spec["rules"].update(rules)
    return Engine(spec)


def _stocked(name: str, **rules: Any) -> Engine:
    """The same ruleset plus an endless supply of stones *and* a rule to win by.

    The one shape the two implementations of the goal weight's **unit** can differ on.
    That unit is the mean value of a piece that is material, and an inexhaustible type is
    not material -- it is priced at ``INEXHAUSTIBLE_VALUE`` and is out of the two scales
    in ``Evaluator.for_spec`` for the same reason. Leaving it in averages a zero with
    every real piece and understates the goal term, and the two sides derive that mean
    separately: Python from ``inexhaustible_types``, the core from
    ``DropRule::is_unlimited``. Both read the same ``drops.unlimited`` array, and a fix
    applied to one of them only would produce a legal, plausible, *different* game.

    Constructed rather than loaded, because the corpus has no such spec and is not the
    place to invent one: ``archive/stones`` has the endless supply and is won by
    checkmate, so the goal weight it derives is never read. Go is exactly a game with
    both.

    **The two squares are load-bearing** and so is the goal rule being territorial.
    Stones droppable anywhere put twenty-four extra moves in every list, and a search at
    150 nodes then spends its whole budget on the first ply and stops noticing the goal
    term at all -- measured: a 20% change in the weight moved no game on any of the seven
    goal rules tried. A term too small to change a move is a term that would not notice
    the two sides disagreeing about it, which is the same trap the scoring arm's clock
    was. `test_the_goal_weights_unit_changes_this_arms_game` is the standing guard.
    """
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = 10
    spec["pieces"].append({"id": "S", "name": "Stone", "value": 1, "moves": []})
    spec["rules"].update(rules)
    spec["rules"]["drops"] = {
        "from_captures": False,
        "stock": {"S": "unlimited"},
        "region": {"S": ["a3", "b3"]},
    }
    return Engine(spec)


def _stocked_goal_arm() -> Engine:
    """The arm itself, so the boundary check and its guard cannot drift apart."""
    return _stocked("gardner", territory={"mode": "permanent", "win_at": 8})


def _diluted_goal(engine: Engine) -> float:
    """The ``goal=`` that reproduces the weight before an endless type left the mean.

    Not a magic number: the mean over every non-royal type, including the zero, against
    the mean the evaluator now takes. On this arm that is 62.00 against 74.40.
    """
    values = Evaluator.for_spec(engine.spec).values
    n_non_royal = len(values) - len(engine.spec.royal_types)
    diluted = GOAL_PIECES * sum(values) / n_non_royal
    return GOAL_PIECES * diluted / Evaluator.for_spec(engine.spec).goal_weight


def _unwinnable(name: str) -> Engine:
    """The same ruleset plus a stone that nothing able to move may capture.

    The one shape the *prey share*'s denominator can be wrong on, and the corpus reaches
    it in two archived specs and one probe -- none of which is in this file's corpus, and
    two of which are archived precisely because they are broken games. So it is built.

    The share is "of the enemy material anybody could ever win, what fraction may I win",
    and a piece nobody may take is in neither half of it. Counting it in the denominator
    alone made **placing** one cut the value of the opponent's whole army: measured at
    the root of ``archive/stones`` ``93b02b591c87``, one stone was worth **+3.1954** to
    the mover with mobility zeroed, against a pawn of 1.0.

    **Every detail of the construction is load-bearing.**

    *The real pieces name each other and not the stone*, which is what makes a capture
    matrix at all. *The stone's own ``captures`` list is left absent*, which means
    everything -- so ``capture_allowed[stone][stone]`` holds, and the stone's column is
    **not** empty. An implementation that took "no attacker may take this type" off the
    matrix alone would find the stone winnable, by a stone that has no moves, and this
    arm would go on paying the bounty. What is true of the stone is that it can execute
    no capture, and that is the test `prey_types` makes.

    *The stock is finite*, so ``INEXHAUSTIBLE_VALUE`` does not price the stone at zero
    and the numerator is live too: a correct drop pays exactly the stone's own value and
    nothing for the dilution. *The region is two squares*, for the reason ``_stocked``
    gives -- droppable anywhere puts twenty-four extra moves in every list and a
    150-node search then never leaves the first ply.

    **And a stone stands in the opening**, in the c-file pawn's place on both sides, which
    is what makes the boundary comparison a check rather than a hope. The denominators
    only differ once one is on the board, so an arm that relied on the *search* to drop
    one would agree about a branch it might never take -- and a correct drop is now worth
    a fifth of a pawn, which is exactly the kind of move a 150-node search can decline.
    Placed by swapping a pawn rather than added beside one, because a 5x5 opening has no
    free square and rank 3 mirrors onto itself.
    """
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = 10
    real = [piece["id"] for piece in spec["pieces"]]
    spec["pieces"].append({"id": "S", "name": "Stone", "value": 1, "moves": []})
    for piece in spec["pieces"]:
        if piece["id"] != "S":
            piece["captures"] = real
    for placement in spec["setup"]["placements"]:
        if placement["piece"] == "P":
            placement["squares"] = [s for s in placement["squares"] if s != "c2"]
    spec["setup"]["placements"].append({"piece": "S", "squares": ["c2"]})
    spec["rules"]["drops"] = {
        "from_captures": False,
        "stock": {"S": 2},
        "region": {"S": ["a3", "b3"]},
    }
    return Engine(spec)


def _restricted_captures(name: str) -> Engine:
    """The same ruleset with a capture matrix, which is what makes values relational.

    A cycle rather than a hierarchy, because a cycle is the case that cannot be folded
    into a static ordering: each piece is prey for exactly one other, so what a piece is
    worth depends on which army is still standing rather than on a rank.
    """
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = 3
    cycle = {"N": ["B", "P"], "B": ["R", "P"], "R": ["N", "P"]}
    for piece in spec["pieces"]:
        if piece["id"] in cycle:
            piece["captures"] = cycle[piece["id"]]
    return Engine(spec)


# Every rule stage 17A gave the evaluator a term for, plus the one that made piece
# values relational. Built from a spec rather than loaded from `rulesets/` for the same
# reason `screen` and `turn` are: the committed rulesets that declare these are the six
# RPS probes, and a probe is not in this file's corpus -- so without constructing the
# arms deliberately there would be no boundary check on any of it. Gardner because the
# Python side is the slow one and a 5x5 board reaches a territorial win in a few plies.
GOAL_ARMS: list[tuple[str, Any]] = [
    ("territory-permanent", lambda: _with_rules(
        "gardner", 4, territory={"mode": "permanent", "win_at": 8})),
    ("territory-repaint", lambda: _with_rules(
        "gardner", 4, territory={"mode": "repaint", "win_at": 8})),
    # A clock the game actually reaches, as well as a scoring rule. The scoring term is
    # scaled by how close the clock is, because that is when the rule fires -- so on
    # gardner's own 200-ply limit it is worth almost nothing for the twenty plies a 5x5
    # game lasts, and an arm too small to change a move is an arm that would not notice
    # the two sides disagreeing about it. Twelve plies is a game decided on the count.
    ("scoring-material", lambda: _with_rules(
        "gardner", 3, scoring={"count": "material"}, max_plies=12)),
    ("scoring-on-ground", lambda: _with_rules(
        "gardner", 4,
        territory={"mode": "permanent"},
        scoring={"count": "territory"})),
    ("capture-matrix", lambda: _restricted_captures("gardner")),
    # Phase B's three. Same construction and the same reason: these rules *do* have
    # committed rulesets, but they are big ones -- `racingkings` is a 300-ply game and
    # the Python side of this comparison is the slow one -- so the arms are built on a
    # 5x5 board where a race is four moves and a game is twenty plies.
    # A *rook* rather than the king, and that is the whole point of the arm. A reach
    # term on a royal piece rewards marching it forward with nothing opposing it, because
    # `ROYAL_VALUE` is zero -- so the evaluator declines to put a term on one, and an arm
    # that raced the king would be testing a branch that no longer exists.
    ("reach-a-square", lambda: _with_rules(
        "gardner", 2, reach={
            "piece": "R", "squares": ["c3"], "mirror": False, "result": "win"})),
    ("check-limit", lambda: _with_rules("gardner", 2, check_limit=2)),
    # Both directions of the same rule, because they are not the same term: running out
    # is a loss for the horde and a *win* for antichess, and an evaluator that got the
    # sign wrong would play one of them backwards while every move stayed legal.
    ("extinction-loses", lambda: _with_rules(
        "gardner", 2, extinction={"pieces": ["P"], "min_count": 1, "result": "loss"})),
    ("extinction-wins", lambda: _with_rules(
        "gardner", 2, extinction={"pieces": ["P"], "min_count": 1, "result": "win"})),
    # The goal weight's *unit*, rather than any one goal term: a game with both an
    # endless supply and a rule to win by. See `_stocked`.
    ("endless-stock-and-a-goal", _stocked_goal_arm),
    # The prey share's *denominator*, rather than the matrix that fills its numerator: a
    # piece both sides may own and neither may take. See `_unwinnable`.
    ("an-uncapturable-piece", lambda: _unwinnable("gardner")),
]


def _goal_arm_cases() -> list[Any]:
    """Every arm against both pairs of players. The capture matrix's minimax pair is the
    one case over 2 s in the gate (2.1 s, Q749), so it runs in the slow tier; the same arm
    against greedy, and every other arm's minimax pair, stay."""
    cases = []
    for agents in (("minimax:nodes=150", "greedy"), ("minimax:nodes=150", "minimax:nodes=150")):
        for label, build in GOAL_ARMS:
            slow = label == "capture-matrix" and agents[1] != "greedy"
            cases.append(pytest.param(
                label, build, agents, id=f"{agents[0]}-vs-{agents[1]}-{label}",
                marks=[pytest.mark.slow] if slow else [],
            ))
    return cases


@pytest.mark.parametrize(("label", "build", "agents"), _goal_arm_cases())
def test_a_game_won_by_something_other_than_capture_is_searched_the_same_way(
    label: str, build: Any, agents: tuple[str, str]
) -> None:
    """The check stage 17A's evaluator needs, and the reason it is built here.

    Three things now differ between the two implementations and all three are silent.
    The goal terms are one multiplication each and a clamp, and a sign or a
    normalisation out of step produces a legal, plausible, *different* game. The prey
    share is applied per position on both sides rather than folded into the values that
    cross the boundary, which is the one place stage 17A moved a derivation. And the
    search's transposition key is no longer `state.key` on a ruleset with territory --
    two mixes that disagree collide in different places, which changes which move comes
    out of a fraction of positions and nothing else.
    """
    _assert_same_played(build(), agents)


@pytest.mark.parametrize("label,build", GOAL_ARMS, ids=[arm[0] for arm in GOAL_ARMS])
def test_each_constructed_arm_actually_changes_the_game(label: str, build: Any) -> None:
    """An arm that plays gardner's own game is not testing anything.

    The comparison above passes trivially if the rule it was built for never changes a
    move: both sides would agree because neither is doing anything. This is the guard on
    that, and it caught the scoring arm, whose term was scaled to nearly nothing by a
    200-ply clock a 5x5 game never comes close to.
    """
    agents = ("minimax:nodes=150", "greedy")
    assert play_game(build(), agents, seed=5).moves != play_game(
        _engine("gardner"), agents, seed=5
    ).moves


@pytest.mark.parametrize(
    "agents",
    [("minimax:nodes=150", "greedy"), ("minimax:nodes=150", "minimax:nodes=150")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_the_goal_weights_unit_changes_this_arms_game(agents: tuple[str, str]) -> None:
    """What makes the endless-stock arm above a check rather than a ceremony.

    Every other arm in `GOAL_ARMS` is guarded by "does the *rule* change the game", and
    for this one that guard passes for the wrong reason: dropping stones changes gardner
    whatever the goal weight is. What has to change the game here is the **weight**, so
    this plays the same arm twice -- once as the evaluator now takes it, once at the
    weight it took before an inexhaustible type left the mean -- and demands they differ.

    Without this the boundary comparison would keep passing if one side of the boundary
    were fixed and the other were not, because the difference would be too small for a
    150-node search to act on. Measured before the arm was chosen: droppable-anywhere
    stones, or any of seven goal rules on top of them, produced *no* divergence at a 20%
    change in the weight.
    """
    engine = _stocked_goal_arm()
    diluted = f"{agents[0]},goal={_diluted_goal(engine)}"
    assert play_game(engine, agents, seed=5).moves != play_game(
        engine, (diluted, agents[1]), seed=5
    ).moves


# ------------------------------------------------------------------- the race term
#
# `reach_progress` is read only by a weighted player, and `GOAL_ARMS` play the built-in
# objective, which never reaches `FeatureEvaluator` or `features.rs`. So until Q465 no
# boundary check read the term at all -- and both sides read it as exactly zero on a
# `reach` over *any* piece (`piece_type == -1`, which no board byte decodes to), which is
# the rule `intransitive` and `rps-v3` declare. Two implementations that share a gap
# agree, so each arm below comes with a guard that the term changes the game.


def _corner_race(**piece: str) -> Engine:
    """Gardner won by getting a piece to the far corner, `intransitive`'s shape.

    The corner rather than a central square because a pawn one step from a central goal
    wins on move one, and a race decided by the terminal check never consults the term.
    `mirror: "rotate"` is `intransitive`'s, so the two sides race to opposite corners.
    """
    return _with_rules(
        "gardner", 7, reach={"squares": ["e5"], "mirror": "rotate", "result": "win", **piece}
    )


# The same race twice: over any piece, which is the case Q465 fixed, and over a named
# one, which read correctly before it and is held to what it read. A knight rather than
# the rook because on this arm the rook's race does not change a game against greedy.
RACE_ARMS: list[tuple[str, Any]] = [
    ("any-piece", _corner_race),
    ("a-named-piece", lambda: _corner_race(piece="N")),
]

RACE_WEIGHTS = _vector(material=1.0, mobility=0.05, centrality=0.2, reach_progress=0.6)
NO_RACE_WEIGHTS = _vector(material=1.0, mobility=0.05, centrality=0.2)


def _race_agent(weights: tuple[float, ...], name: str) -> str:
    heuristic = Heuristic(name=name, plan="", weights=weights, values=())
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
def test_the_race_term_is_the_built_in_evaluators_distance(label: str, build: Any) -> None:
    """`reach_progress` against an independent reader of the same distance table.

    `Evaluator._nearest_to_goal` is the built-in objective's own walk over
    `goal_distance`, and it has always read `-1` as any piece. The term is that distance
    normalised and signed, so the two must agree at every position of a played game --
    and on the any-piece arm the term read `0.0` everywhere before Q465 while the
    distances it is built from did not.
    """
    from chess_variant_sandbox.agents.features import FeatureEvaluator

    engine = build()
    spec = engine.spec
    reach = spec.reach
    assert reach is not None
    reader = FeatureEvaluator(feature_context(spec), (1.0,) * len(VOCABULARY))
    oracle = Evaluator.for_spec(spec)
    index = VOCABULARY.index("reach_progress")
    diameter = float(spec.goal_diameter)

    def expected(state: Any, player: int) -> float:
        near = [
            1.0 - min(1.0, oracle._nearest_to_goal(state, side, reach.piece_type) / diameter)
            for side in (0, 1)
        ]
        return near[player] - near[1 - player]

    # The positions of a game the race-aware player actually plays, so the racing
    # pieces move and get taken rather than standing where they started.
    record = play_game(engine, (_race_agent(RACE_WEIGHTS, "race-aware"), "greedy"), seed=5)
    state = engine.initial_state()
    nonzero = 0
    for move in record.moves:
        for player in (0, 1):
            value = reader.features(engine, state, player)[index]
            assert value == expected(state, player)
            nonzero += value != 0.0
        state = engine.apply(state, move)
    assert nonzero, "the arm never moved the race, so this asserts nothing"


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
@pytest.mark.parametrize(
    "opponent", ["greedy", "minimax:nodes=150"], ids=lambda name: f"vs-{name}"
)
def test_the_race_term_is_read_the_same_way_on_both_sides(
    label: str, build: Any, opponent: str
) -> None:
    """A player weighting the race, played by the core and by the Python oracle."""
    _assert_same_played(build(), (_race_agent(RACE_WEIGHTS, "race-aware"), opponent))


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
def test_caring_about_the_race_changes_the_game(label: str, build: Any) -> None:
    """The guard on the arm above, and the core half of the fix.

    The two players differ only in `reach_progress`. Before Q465 the core read the term
    as zero on the any-piece arm, so they played one game and the comparison above held
    the two sides to a term neither used.
    """
    engine = build()
    aware = _race_agent(RACE_WEIGHTS, "race-aware")
    blind = _race_agent(NO_RACE_WEIGHTS, "race-blind")
    assert (
        play_game(engine, (aware, "greedy"), seed=5).moves
        != play_game(engine, (blind, "greedy"), seed=5).moves
    )


# ------------------------------------------------------------ the race's own constant
#
# Q469 (F1): the race is a *distance* and takes `RACE_SHARE` of the goal weight, where
# the counting terms take all of it. Both sides split the term out the same way; these
# hold the share, and the built-in objective's reading of it, together across the
# boundary on `intransitive`'s shape. `docs/reference/weak-player-intransitive.md` §9.

# The built-in objective's own players: the default, and `goal=128`, which gives the race
# what the default gave it before the share (16 = 128 / 8).
RACE_SHARE_PLAYERS = ("minimax:nodes=150", "minimax:nodes=150,goal=128")
# A searching opponent rather than greedy: against greedy at seed 5 the named-piece arm's
# knight race plays one game at both weights, which would leave the comparison unguarded.
RACE_SHARE_OPPONENT = "minimax:nodes=150"


def test_the_race_share_is_the_cores() -> None:
    """One constant, stated on both sides of the boundary and held together here."""
    assert _core.RACE_SHARE == RACE_SHARE


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
def test_the_race_at_the_default_is_what_goal_2_gave_it(label: str, build: Any) -> None:
    """The share is a power of two so this is exact, not approximate.

    What makes the claim "hardest now plays as the `goal=2` player did" checkable move
    for move on a ruleset whose only goal term is the race, as `intransitive`'s is.
    """
    spec = build().spec
    assert Evaluator.for_spec(spec).race_weight == Evaluator.for_spec(
        spec, goal_pieces=2.0
    ).goal_weight
    assert Evaluator.for_spec(spec, goal_pieces=0.0).race_weight == 0.0


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
@pytest.mark.parametrize("player", RACE_SHARE_PLAYERS, ids=lambda name: name)
def test_the_built_in_race_is_weighed_the_same_way_on_both_sides(
    label: str, build: Any, player: str
) -> None:
    """The built-in objective racing, played by the core and by the Python oracle."""
    _assert_same_played(build(), (player, RACE_SHARE_OPPONENT))


@pytest.mark.parametrize("label,build", RACE_ARMS, ids=[arm[0] for arm in RACE_ARMS])
def test_the_race_share_changes_the_game(label: str, build: Any) -> None:
    """The guard on the comparison above: the share is a weight a search acts on here.

    Without it the comparison would keep passing with the share applied on one side of
    the boundary only, because neither side's moves would depend on it.
    """
    engine = build()
    games = [
        play_game(engine, (player, RACE_SHARE_OPPONENT), seed=5).moves
        for player in RACE_SHARE_PLAYERS
    ]
    assert games[0] != games[1]


# ----------------------------------------------------------------------- the hand
#
# `v11` A6: a piece in hand is material at its full value, on both sides of the
# boundary. `CORPUS` declares no `drops`, so before these arms no played game compared
# the two implementations with anything in a hand -- the endless-stock arm's stone is
# priced at zero and never reaches the term. Three shapes: a committed `from_captures`
# game on the smallest board there is; the same rule on gardner, where promotion and
# capture both feed the hand; and a finite stock, whose hands are full at ply zero.


def _finite_stock(name: str) -> Engine:
    """The same ruleset with one knight a side in hand from the opening."""
    spec: dict[str, Any] = load_spec(RULESETS / f"{name}.json")
    spec["schema_version"] = 10
    spec["rules"]["drops"] = {"from_captures": False, "stock": {"N": 1}}
    return Engine(spec)


HAND_ARMS: list[tuple[str, Any]] = [
    ("dobutsu", lambda: _engine("dobutsu")),
    ("from-captures-on-gardner", lambda: _with_rules(
        "gardner", 2, drops={"from_captures": True})),
    ("a-finite-stock", lambda: _finite_stock("gardner")),
]


def _hand_vector(engine: Engine, hand: bool) -> str:
    """`builtin_vector` as a registered player, with or without its hand weight.

    Without it is the built-in objective of the same rules read as holding no hand --
    the shape before A6, Q228's `h = 0` arm -- which `builtin_vector` builds at unit
    length itself. Zeroing the hand slot of the unit vector instead left the same
    direction at about 0.707 of unit length: on gardner, the digest of gardner's own
    built-in, `2a0d1025586e`, spelled with other raw numbers, which the core plays as
    another player (Q431) and registration now refuses (Q433). Built this way it is
    gardner's own built-in to the bit, one player under one digest.
    """
    context = feature_context(engine.spec)
    if not hand:
        context = replace(context, has_drops=False)
    heuristic = Heuristic(
        name=f"builtin-{'with' if hand else 'without'}-hand",
        plan="",
        weights=builtin_vector(context),
        values=(),
    )
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


@pytest.mark.parametrize(
    "label,build",
    [arm for arm in HAND_ARMS if arm[0] != "dobutsu"],
    ids=[arm[0] for arm in HAND_ARMS if arm[0] != "dobutsu"],
)
def test_the_arm_without_the_hand_is_gardners_own_built_in(label: str, build: Any) -> None:
    """The gardner arms' hand-free player shares gardner's digest, so it must be gardner's.

    Registered after gardner's own built-in, as `test_valuations.py`'s null arm registers
    it: a second spelling of that digest raises here rather than making whichever of the
    two a worker met first the player both names mean.
    """
    plain = Heuristic("plain", "", builtin_vector(feature_context(_engine("gardner").spec)))
    register_heuristic(plain)
    assert _hand_vector(build(), hand=False) == f"minimax:h={plain.digest},nodes=150"


@pytest.mark.parametrize("label,build", HAND_ARMS, ids=[arm[0] for arm in HAND_ARMS])
@pytest.mark.parametrize(
    "agents",
    [("minimax:nodes=150", "greedy"), ("minimax:nodes=150", "minimax:nodes=150")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_a_piece_in_hand_is_counted_the_same_way_on_both_sides(
    label: str, build: Any, agents: tuple[str, str]
) -> None:
    """The built-in objective with a hand, and the recorded material series with it.

    `material` is compared ply by ply as well as the moves, and since A6 each entry is
    board plus hand, so a hand valued differently on the two sides fails here on the
    first ply that holds anything, whether or not a search noticed.
    """
    _assert_same_played(build(), agents)


@pytest.mark.parametrize("label,build", HAND_ARMS, ids=[arm[0] for arm in HAND_ARMS])
def test_the_vocabulary_spells_the_hand_the_same_way_on_both_sides(
    label: str, build: Any
) -> None:
    """`builtin_vector` on a hand game, through `FeatureEvaluator` on both sides.

    Its hand weight is what makes it the built-in objective there, and the column it
    weights is `Evaluator.hand_value` over `total_material` on both sides.
    """
    engine = build()
    _assert_same_played(engine, (_hand_vector(engine, hand=True), "greedy"))


@pytest.mark.parametrize("label,build", HAND_ARMS, ids=[arm[0] for arm in HAND_ARMS])
def test_each_hand_arm_holds_something_and_the_hand_changes_its_game(
    label: str, build: Any
) -> None:
    """An arm whose hands stay empty, or whose hand no search acts on, tests nothing.

    Two guards. The played game passes through positions where the hands are worth
    something, so the term is read with a non-zero value. And the built-in's vector
    with its hand weight plays a different game from the same vector without it, so at
    150 nodes the term changes a move -- which is what makes agreement above a check
    rather than a ceremony.
    """
    engine = build()
    evaluator = Evaluator.for_spec(engine.spec, mobility_weight=0.0)
    record = play_game(engine, ("minimax:nodes=150", "minimax:nodes=150"), seed=5)
    state = engine.initial_state()
    held = 0
    for move in record.moves:
        held += any(
            count and evaluator.values[piece_type]
            for side in state.hands
            for piece_type, count in enumerate(side)
        )
        state = engine.apply(state, move)
    assert held > 0, "the hands never held a priced piece"

    # Against a searcher: `greedy` ends dobutsu in seven plies, before any hand matters.
    opponent = "minimax:nodes=150"
    with_hand = play_game(engine, (_hand_vector(engine, hand=True), opponent), seed=5)
    without = play_game(engine, (_hand_vector(engine, hand=False), opponent), seed=5)
    assert with_hand.moves != without.moves


def _by_army(engine: Engine, position: bytes) -> float:
    """`_relational_material` for white, with every enemy piece back in the denominator.

    The superseded reading, written out rather than reached by a flag, for the same
    reason `_diluted_goal` computes the superseded goal weight rather than restoring it:
    a constant a test can switch on is a constant a player could be built with.
    """
    spec = engine.spec
    values = Evaluator.for_spec(spec, mobility_weight=0.0).values
    side_stride = spec.n_facings * spec.n_types
    counts = [[0] * spec.n_types, [0] * spec.n_types]
    for value in position:
        if value:
            counts[(value - 1) // side_stride][(value - 1) % spec.n_types] += 1
    total = 0.0
    for value in position:
        if value == 0:
            continue
        owner, piece_type = (value - 1) // side_stride, (value - 1) % spec.n_types
        enemy = counts[1 - owner]
        army = sum(enemy)
        allowed = sum(
            count
            for victim, count in enumerate(enemy)
            if spec.capture_allowed[piece_type][victim]
        )
        share = max(allowed / army, 0.1) if army else 1.0
        worth = values[piece_type] * share
        total += worth if owner == 0 else -worth
    return total


def test_an_uncapturable_piece_is_not_the_enemys_army_to_discount() -> None:
    """What makes the uncapturable-stone arm a check rather than a ceremony.

    The `GOAL_ARMS` guard asks "does the *rule* change the game", and for this arm that
    passes for the wrong reason: stones change gardner whether or not the denominator is
    right. What has to be visible here is the **denominator**, so this prices one drop
    both ways -- mobility zeroed, which is how the defect was measured -- and demands the
    difference be bigger than a whole piece, the unit `analysis` calls a distinguishable
    score.

    It also pins the shape the arm exists for, because that is the part a plausible fix
    misses. The stone's own row is all true, so its column is not empty and "no attacker
    may take this type" is *false* of it; what is true is that no attacker that **can
    move** may take it. A pool read off the matrix alone would leave this arm paying, and
    the boundary comparison above would go on passing with both sides wrong together.
    """
    engine = _unwinnable("gardner")
    spec = engine.spec
    evaluator = Evaluator.for_spec(spec, mobility_weight=0.0)
    stone = next(t for t in range(spec.n_types) if not evaluator.prey[t])

    assert spec.has_capture_matrix
    assert spec.capture_allowed[stone][stone], "the trap this arm is built around"
    assert not any(
        spec.capture_allowed[attacker][stone]
        for attacker in range(spec.n_types)
        if attacker != stone
    )

    board = bytes(spec.initial_board)
    dropped = bytearray(board)
    dropped[board.index(0)] = stone + 1  # white, first frame, first empty square
    pool = evaluator.board_material(bytes(dropped), 0) - evaluator.board_material(board, 0)
    army = _by_army(engine, bytes(dropped)) - _by_army(engine, board)

    assert pool == pytest.approx(evaluator.values[stone]), (
        "a drop is worth the piece dropped and not a share of the opponent's army"
    )
    assert army - pool > evaluator.cheapest_piece(board), (
        "the arm cannot see the difference it was built for"
    )


def test_the_uncapturable_arm_has_one_on_the_board_from_the_opening() -> None:
    """And it has to be there from ply zero, or the comparison may never reach it.

    The two denominators agree exactly while no uncapturable piece stands, so an arm that
    left it to the *search* to drop one would let the boundary comparison agree about a
    branch it never took -- and after the fix a drop is worth a fifth of a pawn, which is
    exactly the sort of move a 150-node search declines. So the opening carries one a
    side, and this is the standing guard on that: every position the comparison visits
    prices the pieces differently under the two readings, by more than the cheapest piece
    on the board.
    """
    engine = _unwinnable("gardner")
    spec = engine.spec
    evaluator = Evaluator.for_spec(spec, mobility_weight=0.0)
    stone = next(t for t in range(spec.n_types) if not evaluator.prey[t])
    board = bytes(spec.initial_board)

    n_types = spec.n_types
    standing = [value for value in board if value and (value - 1) % n_types == stone]
    assert len(standing) == 2, "one uncapturable piece a side, in the opening"

    pooled = evaluator._live_values(board)[0]
    side_stride = spec.n_facings * n_types
    counts = [0] * n_types
    for value in board:
        if value and (value - 1) // side_stride == 1:
            counts[(value - 1) % n_types] += 1
    gap = max(
        abs(
            pooled[attacker]
            - evaluator.values[attacker]
            * max(
                sum(
                    count
                    for victim, count in enumerate(counts)
                    if spec.capture_allowed[attacker][victim]
                )
                / sum(counts),
                0.1,
            )
        )
        for attacker in range(n_types)
    )
    assert gap > evaluator.cheapest_piece(board), (
        "the opening prices every piece the same under both readings"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "turn", [{"moves": 2}, {"moves": 2, "ramp": [1]}], ids=["two-moves", "ramped"]
)
def test_the_tree_search_also_handles_a_side_that_moves_twice(turn: dict[str, Any]) -> None:
    """MCTS carries the same assumption in two other places.

    UCT's exploitation term is ``1 - winrate`` because a child belongs to the opponent,
    and backpropagation flips the reward at every step towards the root for the same
    reason. Both are false when a player moves again, and both are silent: the tree
    still grows, the rollouts still finish, and the agent simply plays the moves that
    help whoever it thinks is not itself.
    """
    _assert_same_played(_turned("gardner", **turn), ("mcts:nodes=30", "random"))


def test_one_move_at_a_time_is_the_same_player() -> None:
    """``choose_move`` and the game loop must not be two different players.

    The interactive opponent and the analysis probe ask for one decision at a time. If
    that answered differently from the agent inside a played game, a person would be
    playing something other than the thing the index rated.
    """
    from chess_variant_sandbox.agents import build_agent
    from chess_variant_sandbox.tournament.play import choose_move, seat_rng

    engine = _engine("gardner")
    state = engine.initial_state()
    for ply in range(12):
        moves = engine.legal_moves(state)
        if not moves:
            break
        agent = "minimax:nodes=200" if ply % 2 else "greedy"
        native = choose_move(engine, agent, state, seed=ply)
        oracle = build_agent(agent, engine).select_move(state, moves, seat_rng(ply, 0))
        assert native == oracle
        state = engine.apply(state, native)


# ------------------------------------------------------- a player's piece values
#
# Stage 30 lets a heuristic say what the pieces are worth, and the override crosses the
# boundary raw: it is rescaled to the derived table's mean on both sides, and it is
# substituted into `material` *before* the live prey share multiplies it. Three things
# that have to agree to the bit, none of which any other arm reaches.

VALUED_WEIGHTS = _vector(material=1.0, mobility=0.05)


def _priced(engine: Engine, values: tuple[float, ...] | None, name: str) -> str:
    """An agent whose objective prices the pieces as told, or leaves them derived.

    Registered rather than constructed inline because a heuristic is resolved by digest
    and the digest now covers the values -- which is the point: two spellings of one
    valuation are one player, and two valuations are two.
    """
    heuristic = Heuristic(
        name=name, plan="", weights=VALUED_WEIGHTS, values=tuple(values or ())
    )
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


def _derived_values(engine: Engine) -> tuple[float, ...]:
    return Evaluator.for_spec(engine.spec).values


def _spread_values(engine: Engine) -> tuple[float, ...]:
    """The derived table fanned out, so later piece types are worth relatively more.

    A reallocation rather than a rescale, and deliberately not a reversal: the three RPS
    probes give every piece the same moves, so their derived tables are `(1, 1, 1)` and
    reversing one is the identity. Those are the rulesets a values override is *most*
    interesting on -- the pieces differ only in what they may capture, which is exactly
    what reach cannot see -- so the perturbation has to be one that moves a flat table.

    Royal slots stay at zero, because a non-zero one is refused; that is its own arm.
    """
    royal = set(engine.spec.royal_types)
    return tuple(
        0.0 if index in royal else value * (index + 1)
        for index, value in enumerate(_derived_values(engine))
    )


# One ruleset from each branch of `Evaluator.material`, because the override is
# substituted inside all three and they share no arithmetic: `rps-v1` scales each piece
# by the share of the *live* enemy army it may take, `antichess` flips the sign of the
# whole term, and `chess` is the plain signed count.
VALUE_ARMS = ["chess", "rps-v1", "rps-v2", "antichess", "xiangqi"]


@pytest.mark.parametrize("name", VALUE_ARMS)
def test_an_override_equal_to_the_derived_table_changes_no_bit(name: str) -> None:
    """Inertness, on a ruleset from each of the three `material` branches.

    The rescale is `mean(|override|) / mean(|derived|)`, so an override that *is* the
    derived table divides a float by itself and multiplies by exactly 1.0. That has to
    be true rather than nearly true: it is what makes "this player has no opinion" and
    "this player agrees with the rules" the same player, and it is the property the
    whole scale convention rests on.
    """
    engine = _engine(name)
    plain = _priced(engine, None, f"{name}-plain")
    inert = _priced(engine, _derived_values(engine), f"{name}-inert")
    first = play_game(engine, (plain, "random"), seed=5)
    second = play_game(engine, (inert, "random"), seed=5)
    assert first.moves == second.moves
    assert first.nodes == second.nodes
    assert first.material == second.material


@pytest.mark.parametrize("name", VALUE_ARMS)
@pytest.mark.parametrize(
    "opponent", ["greedy", "minimax:nodes=150"], ids=lambda o: f"vs-{o}"
)
def test_a_priced_objective_is_searched_the_same_way_on_both_sides(
    name: str, opponent: str
) -> None:
    """What holds the two implementations to the bit once a player prices the pieces.

    Three substitutions, all silent if one side does them and the other does not: the
    rescale to the derived mean, the values `material` sums in board order, and -- on
    `rps-v1` and `rps-v2` -- the live prey share, which multiplies the *override* rather
    than the derived table because the share is a correction to a valuation.
    """
    engine = _engine(name)
    _assert_same_played(engine, (_priced(engine, _spread_values(engine), name), opponent))


@pytest.mark.parametrize("name", VALUE_ARMS)
def test_each_priced_arm_actually_changes_the_game(name: str) -> None:
    """An arm that plays the derived valuation's game is not testing anything.

    The comparison above would pass trivially if the override never changed a move --
    both sides would agree because neither is doing anything.
    """
    engine = _engine(name)
    plain = play_game(engine, (_priced(engine, None, f"{name}-plain"), "greedy"), seed=5)
    priced = play_game(
        engine, (_priced(engine, _spread_values(engine), name), "greedy"), seed=5
    )
    assert plain.moves != priced.moves


def test_the_core_refuses_a_table_written_for_another_game() -> None:
    """The boundary is where a bad override stops, and it stops with a reason.

    `Player::new` builds the pair inside a detached thread, where there is nothing to
    return an error to, so `AgentSpec::check_against` runs at the conversion and every
    path to a player comes through it. A truncated or reinterpreted table would be a
    differently-playing agent under an unchanged name.
    """
    from chess_variant_sandbox.agents import native_spec

    engine = _engine("chess")
    agent = native_spec("minimax:nodes=150", engine.spec.hash)
    assert agent is not None
    wrong = (*agent[:7], VALUED_WEIGHTS, (1.0, 2.0), *agent[9:])
    with pytest.raises(ValueError, match="different game"):
        engine.core.select_move(_pack(engine.initial_state()), wrong, [1.0], 0)


def test_the_core_names_both_lengths_for_a_vector_from_another_vocabulary() -> None:
    """Q142: the refusal names the check that failed, never the agent kind.

    Every refusal in `AgentSpec::from_parts` used to reach Python as "minimax has no
    native implementation", so a searched table stored under the previous vocabulary
    -- one weight short -- sent a person looking for a missing agent. The length is the
    whole diagnosis, so both lengths are in the message.
    """
    from chess_variant_sandbox.agents import native_spec

    engine = _engine("chess")
    agent = native_spec("minimax:nodes=150", engine.spec.hash)
    assert agent is not None
    state = _pack(engine.initial_state())
    for length in (len(VOCABULARY) - 1, len(VOCABULARY) + 1):
        foreign = (*agent[:7], (0.1,) * length, (), *agent[9:])
        with pytest.raises(ValueError) as refused:
            engine.core.select_move(state, foreign, [1.0], 0)
        message = str(refused.value)
        assert "weights.len() != VOCABULARY_LEN" in message
        assert f"expected {len(VOCABULARY)}, got {length}" in message
        assert "native implementation" not in message
    # The one refusal that *is* about the kind still says so.
    with pytest.raises(ValueError, match="has no native implementation"):
        engine.core.select_move(state, ("fairystockfish", *agent[1:]), [1.0], 0)


def test_the_core_refuses_a_priced_royal() -> None:
    """`ROYAL_VALUE = 0.0` is load-bearing on this side of the boundary too."""
    from chess_variant_sandbox.agents import native_spec

    engine = _engine("chess")
    agent = native_spec("minimax:nodes=150", engine.spec.hash)
    assert agent is not None
    values = [1.0] * engine.spec.n_types
    values[engine.spec.royal_types[0]] = 3.0
    priced = (*agent[:7], VALUED_WEIGHTS, tuple(values), *agent[9:])
    with pytest.raises(ValueError, match="royal"):
        engine.core.select_move(_pack(engine.initial_state()), priced, [1.0], 0)


# --------------------------------------------- a piece with a direction (schema 8)
#
# `facing_alignment` is the only term in the vocabulary that reads a piece's *frame*,
# and the frame is the one field on a board byte that no other term decodes. The two
# sides work it out independently: Python divides the byte by `n_types` and then splits
# the frame with `//` and `%`, the core does the same off `Tables`. Every ruleset in
# `CORPUS` above has one facing, so every arm in this file compares the two sides only
# on boards where the frame *is* the owner -- which is exactly the set of boards where
# the two readings cannot disagree.
#
# So the arm below is built to be a board they can disagree on, and the test after it
# asserts that it is one rather than assuming it. Three properties, none accidental:
# the two players hold **different** multisets of facings, so reading the frame as the
# owner is not merely wrong but wrong by a different amount for each side; a type that
# turns stands beside four that cannot, so `facing_axes` is exercised rather than
# defaulted; and the term reads non-zero at ply zero, so an arm that quietly returned
# `0.0` on both sides would fail rather than pass.

_FACING_FILES = "abcde"
_FACING_FRONT = {0: "2", 1: "4"}
_FACING_BACK = {0: "1", 1: "5"}

# One facing per pawn, per player, and the two rows are deliberately *not* a permutation
# of each other: with the same multiset on both sides a frame-as-owner misread would
# cancel in the subtraction and the arm would agree with a bug.
_FACINGS = {
    0: ("forward", "right", "backward", "left", "forward"),
    1: ("backward", "left", "forward", "right", "right"),
}


def _oriented() -> Engine:
    """Gardner with a pawn that can turn, and every pawn pointed somewhere different.

    Built rather than loaded for the reason `GOAL_ARMS` are: the two committed rulesets
    that declare a facing are `chess-oriented-pawns`, whose pawns all start pointing the
    same way, and `archive/weathervane.json`, which is an 8x8 board with sixty-four
    pieces on it and a 400-ply clock -- and the Python side of this comparison is the
    slow one. A 5x5 board with a rolled facing is the same question in twenty plies.

    `mirror: false` because the placements are the point: a mirrored setup gives the two
    armies the same facings by construction, which is the one thing this arm must not do.
    """
    spec: dict[str, Any] = load_spec(RULESETS / "gardner.json")
    spec["schema_version"] = 8
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            piece["facing"] = {"axes": 4}
    placements: list[dict[str, Any]] = []
    for player in (0, 1):
        for piece, file in zip("RNBQK", _FACING_FILES, strict=True):
            placements.append(
                {"piece": piece, "player": player, "squares": [file + _FACING_BACK[player]]}
            )
        for facing in ("forward", "right", "backward", "left"):
            squares = [
                _FACING_FILES[index] + _FACING_FRONT[player]
                for index, name in enumerate(_FACINGS[player])
                if name == facing
            ]
            if squares:
                placements.append(
                    {"piece": "P", "player": player, "squares": squares, "facing": facing}
                )
    spec["setup"] = {"mirror": False, "placements": placements}
    return Engine(spec)


# The same player twice, with and without an opinion about which way a piece points.
# `advance` is in both so that the arms are two readings of one game rather than one
# player that cares about geometry and one that does not.
FACING_WEIGHTS = _vector(material=1.0, mobility=0.05, advance=0.3, facing_alignment=0.6)
BLIND_WEIGHTS = _vector(material=1.0, mobility=0.05, advance=0.3)


def _facing_agent(weights: tuple[float, ...], name: str) -> str:
    heuristic = Heuristic(name=name, plan="", weights=weights, values=())
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


@pytest.mark.parametrize(
    "agents",
    [("minimax:nodes=150", "greedy"), ("minimax:nodes=150", "minimax:nodes=150")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_a_board_that_faces_is_searched_the_same_way_on_both_sides(
    agents: tuple[str, str],
) -> None:
    """The movegen half: a game with rotations in it, played by the built-in objective.

    Nothing here reads `facing_alignment` -- the built-in evaluator never reaches
    `features.rs` -- and that is the point of having it separately from the arm below.
    A turn is a move that changes a board byte's frame and nothing else, so a side that
    applied or undid one differently would produce a legal, plausible, different game.
    """
    _assert_same_played(_oriented(), agents)


@pytest.mark.parametrize(
    "agents",
    [("greedy", "greedy"), ("minimax:nodes=150", "greedy")],
    ids=lambda pair: f"{pair[0]}-vs-{pair[1]}",
)
def test_the_facing_term_is_read_the_same_way_on_both_sides(
    agents: tuple[str, str],
) -> None:
    """The arm that holds `facing_alignment` to the bit across the boundary.

    A weighted player is the only thing that reaches `FeatureEvaluator` at all, so
    without this the fifteenth term is outside the boundary check entirely -- and it is
    the one term whose decode has no second reader anywhere in the suite.

    **It failed the first time it was run, and not on the term it was written for.**
    `agents/minimax.py` had never been taught schema 8: `_is_loud` read a rotation's
    target square, which holds the piece that is turning, so the Python oracle called
    every turn a capture and chased them all through quiescence; and `_capture_gain` read
    the same square and ordered every turn ahead of every quiet move by the value of the
    piece doing the turning. `agents.rs` guards both by name and has since schema 8. The
    two were a different player on any board that turns, silently, and nothing saw it
    because nothing had ever played the Python side on one. Pinned directly by
    `test_the_python_oracle_knows_a_rotation_takes_nothing` below.
    """
    engine = _oriented()
    agent = _facing_agent(FACING_WEIGHTS, "facing-aware")
    _assert_same_played(engine, (agent, agents[1]))


def test_the_python_oracle_knows_a_rotation_takes_nothing() -> None:
    """A turn ends where it began, so its target square is never empty.

    Both halves of what the boundary arm above caught, asserted where they live rather
    than through a whole game: quiescence must not chase a turn, and move ordering must
    not price one by the piece that is turning. `agents.rs::is_loud` and
    `agents.rs::capture_gain` each guard this by name, and this is the same guard on the
    oracle. On `weathervane` most of a move list is rotations, so getting either wrong is
    not a tie-break -- it is a different search.
    """
    from chess_variant_sandbox.agents import build_agent
    from chess_variant_sandbox.engine.moves import MOVE_ROTATE, move_to, move_type

    engine = _oriented()
    state = engine.initial_state()
    agent = build_agent("minimax:nodes=150", engine)
    turns = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_ROTATE]
    assert turns, "the arm has no rotations in it, so this asserts nothing"
    for move in turns:
        # The trap: the square is occupied, by the piece doing the turning.
        assert state.board[move_to(move)] != 0
        assert not agent._is_loud(state, move)
        assert agent._capture_gain(state.board, move) == 0.0


def test_the_facing_arm_is_a_position_a_wrong_reading_would_get_wrong() -> None:
    """The guard, and it is the whole of what the brief for this term asked for.

    Two implementations that share a gap *agree*, so a byte-identity check is only worth
    what its position is worth. This asserts the position is one where the two mistakes
    anybody would actually make -- reading the frame as the owner, and reading the facing
    off the raw byte instead of off the frame -- each produce a **different number** from
    the right answer. Both wrong readings are written out here rather than described,
    because "a wrong reading would differ" is a claim about arithmetic and this is the
    arithmetic.
    """
    from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator, feature_context

    engine = _oriented()
    spec = engine.spec
    state = engine.initial_state()
    context = feature_context(spec)
    index = VOCABULARY.index("facing_alignment")
    reader = FeatureEvaluator(context, (1.0,) * len(VOCABULARY))
    correct = reader.features(engine, state, 0)[index]

    # The arm is a board with a facing on it, and the two armies do not hold the same
    # facings. Without this the checks below could pass on a board nothing turns on.
    assert spec.n_facings == 4 and spec.has_facing
    assert sorted(spec.facing_axes) == [1, 1, 1, 1, 1, 4]
    held: dict[int, list[int]] = {0: [], 1: []}
    for value in state.board[: spec.n_squares]:
        if value:
            frame = (value - 1) // spec.n_types
            if spec.facing_axes[(value - 1) % spec.n_types] > 1:
                held[frame // spec.n_facings].append(frame % spec.n_facings)
    assert len(set(held[0])) > 1, held
    assert sorted(held[0]) != sorted(held[1]), held

    # A term that reads zero on both sides is a term two implementations agree on for
    # the wrong reason.
    assert correct != 0.0
    assert correct == -reader.features(engine, state, 1)[index]

    align = (1.0, 0.0, -1.0, 0.0)
    step = 4 // spec.n_facings

    def misread(owner_of, facing_of) -> float:  # type: ignore[no-untyped-def]
        total, count = 0.0, 0
        for value in state.board[: spec.n_squares]:
            if value == 0 or spec.facing_axes[(value - 1) % spec.n_types] < 2:
                continue
            total += (
                align[(facing_of(value) % spec.n_facings) * step]
                if owner_of(value) == 0
                else -align[(facing_of(value) % spec.n_facings) * step]
            )
            count += 1
        return total / count if count else 0.0

    def frame_of(value: int) -> int:
        return (value - 1) // spec.n_types

    # The mistake `features.FeatureContext`'s own comment warns about by name.
    assert misread(frame_of, lambda v: frame_of(v) % spec.n_facings) != correct
    # And the other half of it: the facing is a field of the frame, not of the byte.
    assert misread(
        lambda v: (v - 1) // spec.side_stride, lambda v: (v - 1) % spec.n_facings
    ) != correct


def test_caring_which_way_a_piece_points_changes_the_game() -> None:
    """An arm whose extra term never moves a piece is an arm testing nothing.

    Same guard as `test_each_constructed_arm_actually_changes_the_game`, one term down:
    the two players differ only in `facing_alignment`, so if the games matched, the
    comparison above would be holding two implementations to a term neither uses.
    """
    engine = _oriented()
    aware = _facing_agent(FACING_WEIGHTS, "facing-aware")
    blind = _facing_agent(BLIND_WEIGHTS, "facing-blind")
    assert (
        play_game(engine, (aware, "greedy"), seed=5).moves
        != play_game(engine, (blind, "greedy"), seed=5).moves
    )


@pytest.mark.parametrize("name", CORPUS)
def test_the_facing_term_is_exactly_zero_where_nothing_turns(name: str) -> None:
    """The other half of the bump's argument, and it has to be exact rather than small.

    `facing_alignment` was appended to a fixed-length vocabulary, which orphans every
    stored fourteen-weight vector whatever it reads. What says the corpus *replay* owes
    nothing beyond that bookkeeping is this: on a ruleset with no facing the term returns
    `0.0` before any arithmetic runs, so a refit's fifteenth column is identically zero
    and its other fourteen weights come out where they were. Read over a played game
    rather than from the opening, because "nothing turns" has to hold at every position.
    """
    from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator, feature_context

    engine = _engine(name)
    index = VOCABULARY.index("facing_alignment")
    reader = FeatureEvaluator(feature_context(engine.spec), (1.0,) * len(VOCABULARY))
    assert not engine.spec.has_facing
    state = engine.initial_state()
    for ply in range(20):
        moves = engine.legal_moves(state)
        if not moves:
            break
        for player in (0, 1):
            value = reader.features(engine, state, player, len(moves))[index]
            assert value == 0.0 and str(value) == "0.0", (name, ply, player, value)
        state = engine.apply(state, moves[len(moves) // 2])


# ------------------------------------------------------- a third owner (schema 9)
#
# Schema 9 put a piece on the board that neither player owns, and the two sides work out
# who owns a square independently: Python divides the board byte by `side_stride` and
# tests the result, the core walks `Game::all_occupied`. Every committed ruleset has two
# owners, so every arm above compares the two sides only on games where a third owner
# never arises -- which is exactly the set of games where the two readings *cannot*
# disagree. Without the arms below the boundary check is silent on the one thing schema 9
# added, and it would stay silent until the showcase was measured.
#
# What it caught, and what it is a pin against: `FeatureContext::new` is the one walk in
# `features.rs` over `initial_board` rather than over the players' union, and it counted a
# boar into `total_material` and into `counts`. `total_material` is the denominator
# `material`, `prey_weight` and `hand_material` divide by, and `counts` is the piece-count
# `score_ceiling`, so the core scaled six terms against a quantity larger than anything
# either army can hold while Python did not. The `pieces` arm below is where that first
# changes a move -- ply 24, on the old core -- and no perft can see any of it.


def _boar(**rules: Any) -> Engine:
    """Chess plus two wild boars: `docs/archive/plan-schema-6-9.md` stage C's showcase, as
    pre-registered.

    One boar per side of the centre, because a route's mirror image may not be itself.
    ``W`` because chess already has a bishop on ``B``. The boar's own moves are a
    knight's -- what it does in a *player's* hands, never how it walks, which is the
    fixed route below.

    Capped at 40 plies because the Python side is the slow one here and the terms under
    test are read at every node of every search, not at the end.
    """
    spec: dict[str, Any] = load_spec(RULESETS / "chess.json")
    spec["schema_version"] = 9
    spec["pieces"].append(
        {
            "id": "W",
            "name": "boar",
            "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        }
    )
    spec["rules"]["npcs"] = [
        {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
    ]
    spec["rules"]["scoring"] = {"count": "points"}
    spec["rules"]["max_plies"] = 40
    spec["rules"].update(rules)
    return Engine(spec)


def _boar_with_a_capture_matrix() -> Engine:
    """The showcase with schema 3's cycle bolted on.

    The arm that reaches the *relational* branch of material and the live prey share --
    `Evaluator::live_values`, `Evaluator::relational_material` and
    `features::prey_weight` -- none of which chess reaches at all, because none of its
    pieces declares `captures`. Each of the three holds a table with a row per player, so
    the share is the quantity a third owner has no column in.
    """
    spec: dict[str, Any] = load_spec(RULESETS / "chess.json")
    spec["schema_version"] = 9
    spec["pieces"].append(
        {
            "id": "W",
            "name": "boar",
            "value": 3,
            "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
        }
    )
    cycle = {"N": ["B", "P"], "B": ["R", "P"], "R": ["N", "P"]}
    for piece in spec["pieces"]:
        if piece["id"] in cycle:
            piece["captures"] = cycle[piece["id"]]
    spec["rules"]["npcs"] = [
        {"piece": "W", "route": ["c3", "c4", "d4", "d3"], "bounty": 3, "mirror": True}
    ]
    spec["rules"]["scoring"] = {"count": "points"}
    spec["rules"]["max_plies"] = 40
    return Engine(spec)


# Every branch of `score_gap` and `score_ceiling` a wild piece can reach, plus the
# capture matrix. `points` is the rule schema 9 added for the sake of wild pieces and
# reads a tally; `material` and `pieces` count what is standing on the board, which is
# where "is this boar mine, theirs, or nobody's" is actually asked.
WILD_ARMS: list[tuple[str, Any]] = [
    ("wild-points", lambda: _boar()),
    ("wild-scoring-material", lambda: _boar(scoring={"count": "material"})),
    ("wild-scoring-pieces", lambda: _boar(scoring={"count": "pieces"})),
    ("wild-capture-matrix", _boar_with_a_capture_matrix),
]

# Every term that decodes an owner, switched on at once. The default rosters above all
# play the *built-in* objective, which reaches `Evaluator` and never `FeatureEvaluator`,
# so without a weighted arm the whole feature vocabulary -- advance, centrality, the prey
# share, the score gap -- is outside the boundary check on a three-owner board.
WILD_WEIGHTS = _vector(
    material=1.0,
    prey_weight=0.6,
    mobility=0.05,
    advance=0.4,
    centrality=0.3,
    royal_exposure=0.2,
    score_margin=0.5,
    score_urgency=0.3,
)


def _weighted(name: str) -> str:
    heuristic = Heuristic(name=name, plan="", weights=WILD_WEIGHTS, values=())
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


@pytest.mark.parametrize("label,build", WILD_ARMS, ids=[arm[0] for arm in WILD_ARMS])
def test_a_board_with_a_third_owner_is_searched_the_same_way_on_both_sides(
    label: str, build: Any
) -> None:
    """The built-in objective, on a board holding a piece neither player owns.

    `Evaluator::material` is a *difference* between the two players and its defining
    property is antisymmetry: ``f(board, 0) == -f(board, 1)``. A boar counted as "not
    this player" breaks that by making both players poorer by the same piece, and it
    breaks it identically at every node -- so the game stays legal and plausible and
    simply is not the game the other side plays.
    """
    _assert_same_played(build(), ("minimax:nodes=150", "greedy"))


@pytest.mark.parametrize("label,build", WILD_ARMS, ids=[arm[0] for arm in WILD_ARMS])
def test_the_feature_vocabulary_agrees_on_a_board_with_a_third_owner(
    label: str, build: Any
) -> None:
    """The same board, read by the terms a fitted heuristic is regressed on.

    This is the arm that matters most, and the one nothing else in the suite reaches: the
    vocabulary is what `chess-variant-sandbox fit` draws a weight vector against, so a term that reads a
    boar as the enemy would be fitted around rather than noticed. Six of the fourteen
    terms are scaled by `total_material`, which is why an arm weighting only `material`
    would still pass while the context underneath it disagreed.
    """
    _assert_same_played(build(), (_weighted(label), "greedy"))


@pytest.mark.parametrize("label,build", WILD_ARMS, ids=[arm[0] for arm in WILD_ARMS])
def test_each_wild_arm_really_puts_a_third_owner_on_the_board(
    label: str, build: Any
) -> None:
    """An arm that plays chess is not testing anything.

    Both comparisons above pass trivially if the spec has two owners after all -- they
    would agree because there is nothing to disagree about -- and that is the failure
    mode this whole section exists to rule out, since it is the state the suite was
    already in. So: three owners compiled, and boars actually standing on the board.
    """
    from chess_variant_sandbox.engine.compile import WILD

    engine = build()
    spec = engine.spec
    assert spec.n_owners == 3
    wild = [
        square
        for square, value in enumerate(spec.initial_board)
        if value and (value - 1) // spec.side_stride == WILD
    ]
    assert len(wild) == 2, wild


# -------------------------------------------------------------------- the movegen


@pytest.mark.parametrize("name", CORPUS)
def test_the_generator_holds_its_own_invariants(name: str) -> None:
    """Properties a move generator has to satisfy whatever the rules say.

    These replace the ply-by-ply comparison against the second interpreter. They are
    weaker in one specific way -- nothing here would notice both sides of a rule being
    implemented consistently wrong -- and that is what the published perft numbers are
    for.
    """
    engine = _engine(name)
    state = engine.initial_state()

    for _ in range(60):
        moves = engine.legal_moves(state)
        pseudo = engine.pseudo_legal_moves(state)

        # Legality only ever removes moves.
        assert set(moves) <= set(pseudo)
        # The incremental key and the full scan are two definitions of one number.
        assert state.key == engine._compute_hash(state)
        # A verdict cannot depend on whether the caller had a move list in hand.
        assert engine.terminal(state, moves) == engine.terminal(state)
        # In check or not, the answer is about the position, not about the argument.
        assert engine.in_check(state) == engine.in_check(state, state.side_to_move)

        if not moves:
            break
        # A middling move rather than the first: the first is a corner piece for most
        # of these openings, and a whole game of them exercises very little.
        chosen = moves[len(moves) // 2]
        before = state
        board_before = bytes(before.board)
        state = engine.apply(state, chosen)
        # ``apply`` returns a new position and leaves the old one untouched -- the whole
        # reason a state can be handed around, cached, and compared.
        assert bytes(before.board) == board_before
        assert before.key == engine._compute_hash(before)
        assert state.ply == before.ply + 1


@pytest.mark.parametrize("name", CORPUS)
def test_a_position_reached_two_ways_is_one_position(name: str) -> None:
    """Replaying a game's moves must rebuild exactly the positions it visited.

    A record stores moves rather than positions, so every consumer -- the visualiser,
    the analysis pass, the web app -- depends on this. It is also the check that caught
    an atomic explosion clearing squares the incremental hash never saw.
    """
    engine = _engine(name)
    record = play_game(engine, ("greedy", "random"), seed=2)

    state = engine.initial_state()
    keys = []
    for move in record.moves:
        assert move in engine.legal_moves(state)
        state = engine.apply(state, move)
        keys.append(state.key)
        assert state.key == engine._compute_hash(state)

    again = engine.initial_state()
    for index, move in enumerate(record.moves):
        again = engine.apply(again, move)
        assert again.key == keys[index]
    assert bytes(again.board) == bytes(state.board)


def test_threading_does_not_change_a_perft_count() -> None:
    """A count split across threads is a sum, so it must not move with the split.

    Two threads split the root moves as surely as eight did, and eight were more cores
    than a worker of the booked fast suite has (Q377c)."""
    engine = _engine("chess")
    state = engine.initial_state()
    assert engine.perft(state, 4, threads=1) == engine.perft(state, 4, threads=2) == 197_281


def test_a_tournament_is_identical_at_any_worker_count() -> None:
    """The property that makes parallel scheduling safe to take for granted.

    Two workers split the schedule as surely as eight did (Q377c: see the perft test)."""
    from chess_variant_sandbox.tournament import TournamentConfig, run_tournament

    spec: dict[str, Any] = load_spec(RULESETS / "gardner.json")
    config = TournamentConfig(agents=("random", "greedy"), games_per_pairing=2, seed=3)

    serial = run_tournament(spec, TournamentConfig(**{**config.__dict__, "workers": 1}))
    parallel = run_tournament(spec, TournamentConfig(**{**config.__dict__, "workers": 2}))

    assert [record.moves for record in serial.records] == [
        record.moves for record in parallel.records
    ]


def test_a_custom_agent_falls_back_to_the_python_loop() -> None:
    """``register`` has to stay a real extension point.

    An agent the core knows nothing about must still play, which means the Python game
    loop cannot be deleted -- it is the fallback, and it outlives the interpreter that
    used to be confused with it.
    """
    import random

    from chess_variant_sandbox.agents import SearchStats, register
    from chess_variant_sandbox.engine import GameState

    class Lowest:
        def __init__(self, engine: Engine) -> None:
            self.name = "lowest"
            self.stats = SearchStats()

        def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
            self.stats.moves += 1
            return min(moves)

    register("lowest")(lambda engine, options: Lowest(engine))

    engine = _engine("gardner")
    record = play_game(engine, ("lowest", "random"), seed=1)
    assert record.plies > 0
    assert record.agents == ("lowest", "random")


def test_a_drop_is_written_as_a_drop() -> None:
    """The encoding sets a drop's origin equal to its destination.

    Reading the board there finds an empty square, which used to name the piece by
    wrapping -1 around the type count: every crazyhouse and shogi drop was reported as
    the last piece type moving from a square to itself, in records and in the visualiser
    alike. No perft could see it, because the move itself was right.
    """
    from chess_variant_sandbox.engine.moves import MOVE_DROP, move_type
    from chess_variant_sandbox.tournament.play import choose_move

    engine = _engine("crazyhouse")
    state = engine.initial_state()
    for ply in range(80):
        drops = [m for m in engine.legal_moves(state) if move_type(m) == MOVE_DROP]
        if drops:
            for move in drops[:4]:
                text = engine.describe_move(state, move)
                assert "@" in text
                assert text[0] in engine.spec.piece_ids
                assert engine.find_move(state, text) == move
            return
        move = choose_move(engine, "greedy", state, seed=ply)
        assert move is not None
        state = engine.apply(state, move)

    raise AssertionError("no drop became available; the test proved nothing")


# ---------------------------------------------------------------- the guard ----
#
# `bridge.rust_supports` is the one place that can say "the compiler can express this
# and the core cannot". It answers `None` for everything today, which is the truth and
# is also why the guard protects nothing by itself: it is a docstring with a call site.
# The test below turns it into a check that a "no" is acted on rather than dropped:
# `Engine.__init__` raises. That the answer is still yes for every committed ruleset is
# then held by anything that builds an `Engine` for each of them, which is
# `test_schema_two.py::test_every_committed_ruleset_plays_a_game_through`; a test of its
# own here repeated that walk (Q377).


def test_a_ruleset_the_core_cannot_play_is_refused_rather_than_approximated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Being wrong in the permissive direction here is silent.

    A primitive that lands in the compiler before it lands in the core would otherwise
    play a plausible game under rules nobody wrote -- no exception, no wrong answer to
    point at, just a different game. There is no such primitive today, so the refusal
    is staged rather than provoked: what is under test is that `Engine.__init__` still
    asks, and still stops when the answer is no.
    """
    import chess_variant_sandbox.engine.engine as engine_module

    monkeypatch.setattr(
        engine_module, "rust_supports", lambda compiled: "the flux capacitor primitive"
    )
    with pytest.raises(NotImplementedError, match="flux capacitor"):
        Engine(load_spec(RULESETS / "chess.json"))


# ----------------------------------------------------------------- the other guard ----
#
# `rust_supports` above asks what the core can express. `core_boundary_mismatch` asks
# whether the extension actually built on this machine is the one this package goes with.
#
# It is not hypothetical the way the flux capacitor is. Schema 9 added the bounty tally to
# the packed position -- slot 11 went from three fields to four -- and a tree with the new
# Python over a core built before it answered every game page with `IndexError: tuple index
# out of range` from `engine._unpack`, which names neither the build nor the fix. The two
# fakes below are that mismatch in both directions, at a tenth of the size: the same
# opening position with slot 11 one field short, and one field long.


class _CoreOfAnotherVintage:
    """A core that hands back a position of a stated shape. That is the whole boundary."""

    def __init__(self, packed: tuple[Any, ...]) -> None:
        self._packed = packed

    def initial_state(self) -> tuple[Any, ...]:
        return self._packed


def _position_with_slot_eleven(packed: tuple[Any, ...], group: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*packed[:11], group)


def test_the_core_on_this_machine_speaks_this_packages_position() -> None:
    """The check's own false-positive test, and the reason it needs no declaration.

    `core_boundary_mismatch` compares the position the core produced with the one
    `_pack(_unpack(...))` rebuilds from it. That only means anything if the round trip is
    shape-stable for a *current* core -- otherwise the check refuses every build, which is
    a worse failure than the one it is here to name.
    """
    from chess_variant_sandbox.engine.bridge import core_boundary_mismatch
    from chess_variant_sandbox.engine.engine import _pack, _unpack

    engine = Engine(load_spec(RULESETS / "chess.json"))
    assert core_boundary_mismatch(engine.core, _unpack, _pack) is None


def test_a_core_built_before_a_field_was_added_names_the_build_and_the_fix() -> None:
    """Schema 9's actual failure, staged: slot 11 with three fields instead of four.

    What the user saw was `IndexError: tuple index out of range` and nothing else. What
    the check has to produce is that error *plus* the two sentences that would have saved
    the ten minutes -- the core is older than the package, and here is how to rebuild it.
    """
    from chess_variant_sandbox.engine.bridge import core_boundary_mismatch
    from chess_variant_sandbox.engine.engine import _pack, _unpack

    engine = Engine(load_spec(RULESETS / "chess.json"))
    packed = engine.core.initial_state()
    older = _CoreOfAnotherVintage(_position_with_slot_eleven(packed, tuple(packed[11][:3])))

    message = core_boundary_mismatch(older, _unpack, _pack)
    assert message is not None
    assert "IndexError: tuple index out of range" in message
    assert "stale build" in message
    assert "maturin develop --release" in message
    assert "uv sync --extra dev --reinstall-package chess-variant-sandbox" in message


def test_a_core_built_after_a_field_was_added_is_caught_too() -> None:
    """The direction that raises nothing at all, and so would otherwise play on.

    A core *newer* than the package -- an older branch checked out over a fresh build --
    hands back a field the package does not read. Nothing throws: `_unpack` takes the seven
    it knows (five until schema 20's `portal_bar`, six until schema 22's `folds`) and drops
    the eighth on the floor. It is caught because the position that goes
    back in is a field shorter than the one that came out, which is a comparison that
    needs no version number on either side to make.
    """
    from chess_variant_sandbox.engine.bridge import core_boundary_mismatch
    from chess_variant_sandbox.engine.engine import _pack, _unpack

    engine = Engine(load_spec(RULESETS / "chess.json"))
    packed = engine.core.initial_state()
    newer = _CoreOfAnotherVintage(_position_with_slot_eleven(packed, (*packed[11], 7)))

    message = core_boundary_mismatch(newer, _unpack, _pack)
    assert message is not None
    assert "slot 11 carries 9 fields where this package writes 8" in message
    assert "maturin develop --release" in message


def test_a_boundary_mismatch_stops_construction_rather_than_a_later_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of asking in `__init__`: one refusal, before anything can index a tuple.

    Staged the same way the flux capacitor is, and for the same reason -- no build on this
    machine answers yes, so what is under test is that `Engine.__init__` asks and stops.
    """
    import chess_variant_sandbox.engine.engine as engine_module

    monkeypatch.setattr(
        engine_module, "core_boundary_mismatch", lambda core, read, write: "a core of 1997"
    )
    with pytest.raises(RuntimeError, match="a core of 1997"):
        Engine(load_spec(RULESETS / "chess.json"))


# ------------------------------------------------------------- the same guard, tables ----
#
# The position is probed; the tables have to be asked about. `tables.rs::field` raises for a
# key that is missing and ignores one that is extra, so a schema bump that adds a table and
# leaves the packed position alone passes every check above while the core plays the game it
# already knew. `_core.TABLE_KEYS` is the core's own registry of what it reads, and
# `core_table_mismatch` compares it with what `build_tables` writes.


def test_the_core_on_this_machine_reads_every_table_this_package_sends() -> None:
    """The invariant, on the real vocabulary: the two key sets are the same set.

    This is the arm that has to pass on every build, and it is also the only check that
    `keys::ALL` in `rust/src/tables.rs` and the dict `build_tables` returns have not drifted
    apart. A key added to one and not the other fails here rather than in a game.
    """
    from chess_variant_sandbox.engine.bridge import build_tables, core_table_mismatch

    engine = Engine(load_spec(RULESETS / "chess.json"))
    assert core_table_mismatch(build_tables(engine.spec), _core.TABLE_KEYS) is None


def test_a_core_that_never_heard_of_a_table_names_it_and_the_fix() -> None:
    """The failure that is silent today: the package grew a key and the core predates it."""
    from chess_variant_sandbox.engine.bridge import build_tables, core_table_mismatch

    engine = Engine(load_spec(RULESETS / "chess.json"))
    package = dict(build_tables(engine.spec))
    package["gravity"] = [0, 1]

    message = core_table_mismatch(package, _core.TABLE_KEYS)
    assert message is not None
    assert "never reads `gravity`" in message
    assert "stale build" in message
    assert "maturin develop --release" in message


def test_a_core_that_reads_a_table_this_package_stopped_sending_is_caught_too() -> None:
    """The other direction: an older branch checked out over a fresh build.

    `field` would raise for a key the core reaches, but most of them sit behind a flag --
    `explosion_immune` is read only when `has_explosion` -- so a ruleset that does not use
    the rule would construct cleanly and play on. The comparison does not care which branch
    would have run.
    """
    from chess_variant_sandbox.engine.bridge import build_tables, core_table_mismatch

    engine = Engine(load_spec(RULESETS / "chess.json"))
    package = dict(build_tables(engine.spec))
    del package["explosion_immune"]

    message = core_table_mismatch(package, _core.TABLE_KEYS)
    assert message is not None
    assert "reads `explosion_immune`, which this package does not send" in message
    assert "maturin develop --release" in message


def test_a_table_the_core_cannot_read_stops_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asked before the tables cross, because after they cross there is nothing to catch.

    Staged on the real vocabulary with one key struck out, which is exactly what a core
    built before that key existed reports. No build on this machine answers yes, so what is
    under test is that `Engine.__init__` asks and stops.
    """
    monkeypatch.setattr(
        _core, "TABLE_KEYS", tuple(k for k in _core.TABLE_KEYS if k != "entry_promotion")
    )
    with pytest.raises(RuntimeError, match="never reads `entry_promotion`"):
        Engine(load_spec(RULESETS / "chess.json"))


# ------------------------------------------- schema 27: each side's own kinds (Q936)
#
# No committed ruleset says `"per_type": "own"`, so the arm is inline: the record's §1 game,
# two armies of different kinds (`docs/reference/extinction-any-kind.md`). Two readings
# cross: the core's ending over each side's own list (`Game::short_of`), and the
# `extinction_margin` feature, which under "own" reads each side's worst own kind against its
# own opening on both sides (`features.rs`, `agents/features.py`).


def _own_kinds() -> Engine:
    def leap(vector: list[int], symmetry: str) -> list[dict[str, Any]]:
        return [{"kind": "leap", "vectors": [vector], "symmetry": symmetry, "mode": "both"}]

    return Engine({
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
    })


def _margin_player() -> str:
    heuristic = Heuristic(
        name="own-kinds", plan="", weights=_vector(material=1.0, extinction_margin=2.0), values=(),
    )
    register_heuristic(heuristic)
    return f"minimax:h={heuristic.digest},nodes=150"


@pytest.mark.parametrize("agents", [("minimax:nodes=150", "greedy"), ("random", "random")])
def test_running_out_of_ones_own_kind_is_played_the_same_way_on_both_sides(
    agents: tuple[str, str],
) -> None:
    _assert_same_played(_own_kinds(), agents)


def test_the_own_kinds_margin_is_read_the_same_way_on_both_sides() -> None:
    _assert_same_played(_own_kinds(), (_margin_player(), "greedy"))


def test_the_own_kinds_arms_end_by_extinction_and_the_margin_changes_the_game() -> None:
    """Guards on the two arms above: the game they play does end by running out, and the
    feature they compare moves a move."""
    engine = _own_kinds()
    assert play_game(engine, ("random", "random"), seed=5).reason == "extinction"
    weighted = play_game(engine, (_margin_player(), "greedy"), seed=5).moves
    assert weighted != play_game(engine, ("minimax:nodes=150", "greedy"), seed=5).moves
