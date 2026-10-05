"""Stage 17A's acceptance tests: can a player see the win condition it is playing for?

Written before the work, so they are falsifiable, and every one of them was failing when
it was written. "The agent feels better" is not a result; these are the statements that
can be wrong.

The measurements they encode are in ``docs/reference/stage-17-measurements.md``, with the
before-and-after for each. Budgets are small on purpose -- these run in the default suite,
and every one of them separated at the budget it is given.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import _score_roots
from chess_variant_sandbox.tournament.play import play_game

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _engine(name: str) -> Engine:
    return Engine(load_spec(RULESETS / f"{name}.json"))


def _final_state(engine: Engine, record: object) -> object:
    state = engine.initial_state()
    for move in record.moves:  # type: ignore[attr-defined]
        state = engine.apply(state, move)
    return state


# ------------------------------------------------------- E1: two games, two rulesets


def test_two_rulesets_with_different_endings_are_not_the_same_game() -> None:
    """`rps-v2` and `rps-v5` share a board, an army and every movement rule.

    One is won by claiming 41 squares; the other compares ground when the clock runs out.
    Before stage 17A they produced the *same move list, ply for ply*, at every seed
    tested -- two different games played identically, because the player could not tell
    them apart. This is the cheapest possible statement that the agent can see its own
    goal, and it is a strict assertion rather than a threshold.
    """
    v2, v5 = _engine("rps-v2"), _engine("rps-v5")
    agents = ("minimax:nodes=400", "minimax:nodes=400")
    for seed in range(4):
        assert (
            play_game(v2, agents, seed, collect_series=False).moves
            != play_game(v5, agents, seed, collect_series=False).moves
        ), f"rps-v2 and rps-v5 played an identical game at seed {seed}"


# --------------------------------------------------- E2: better than chance at the goal


def test_the_searching_agent_claims_at_least_as_much_ground_as_chance() -> None:
    """`rps-v2` is won by claiming 41 of 81 squares.

    Before stage 17A, self-play at the strongest offered budget ended with 44-62 squares
    claimed against random play's 74-78, and won by colouring 0 times in 6 against
    random's 2. It did not merely fail to pursue territory: it claimed *less* ground than
    chance, because claiming means advancing into contact, the material term reads that as
    risk, and the compensation was invisible. Losing to chance at the game's own win
    condition is the clearest single symptom, and the test is that it stops.
    """
    engine = _engine("rps-v2")
    seeds = range(4)

    def claimed(agent: str) -> tuple[float, int]:
        ground, decisive = 0, 0
        for seed in seeds:
            record = play_game(engine, (agent, agent), seed, collect_series=False)
            state = _final_state(engine, record)
            ground += state.colour.count(1) + state.colour.count(2)  # type: ignore[attr-defined]
            decisive += record.reason == "coloured"
        return ground / len(list(seeds)), decisive

    chance_ground, chance_wins = claimed("random")
    searched_ground, searched_wins = claimed("minimax:nodes=400")

    assert searched_ground >= chance_ground, (
        f"the searcher claimed {searched_ground:.0f} squares on average against chance's "
        f"{chance_ground:.0f}: it is still playing a different game"
    )
    assert searched_wins >= chance_wins


# ------------------------------------------------------------------- E3: inertness


# Digests of a fixed roster on a fixed set of seeds. Two tables, because they carry two
# different promises.
#
# `INERT` is the one the *evaluation* phases rest on: these rulesets declare no rule any
# evaluation term reads, so no term may reach them. Its literals held unchanged from the
# build before stage 17A all the way through phase B -- and then phase C moved every one
# of them, which is correct and was written down in advance. A change to the *search*
# moves everything by construction; that is why phase C was one commit and one campaign.
# What the table still guarantees is that an evaluation term added later cannot touch
# them without this failing.
INERT = {
    "atomic": "abb1e79645b587dc",
    "berolina": "9bdf3c0fda6d98b3",
    "chess": "053bcb933aaaa895",
    "dull": "3d9c05b528f2fa9c",
    "gardner": "6098e10800bd6fba",
    "kingofthehill": "abcfbb010c6fcfee",
    "losalamos": "85ea0a3a055cce9b",
    "makruk": "5d16861d32421942",
    "racingkings": "2fb5ca5d6275239c",
    "shatranj": "de4c4dc2f4e0f47a",
    # Q313, 2026-09-18: re-pinned from `f155e1918b24078d`, and *not* by an evaluation
    # term. The compiler used to walk a piece's direction set in tuple-hash order, which
    # put the interpreter's hash width into xiangqi's slide table; sorting the walk
    # reorders its horse's and elephant's moves, so every agent's tie-break lands
    # elsewhere and the game is a different game. The claim this table makes -- that no
    # evaluation term reaches these rulesets -- is untouched, because the other twelve
    # literals did not move.
    "xiangqi": "0b06955f30a013bd",
}

# `PHASE_B` is the other half, and the pins here are *newer*. These three end on a count
# -- of checks, or of pieces left -- and phase B gave the evaluator a term for each, so
# they were pinned before stage 17A, held through phase A unchanged, and moved when phase
# B landed. The literals below are the v6 ones.
#
# The point of keeping them in the suite is that they were the *prediction*, and it was
# made twice and held twice. When phase B landed, exactly six of these failed and the
# thirteen above did not; three of the six then came back when the measurement said a
# distance term only works where the journey is priced. And this table's own docstring
# said "a phase-C change to the search would move both tables" -- which is exactly what
# happened, so both were re-pinned together in that commit.
PHASE_B = {
    # Moved again in stage 20, deliberately and for one reason: the material term's SIGN
    # now follows an extinction rule that says running out is a win. Phase B gave
    # antichess a goal term for that rule and it was not enough -- a goal term is added to
    # material, so the two argued and material won. Shedding a piece, which is progress
    # towards the win condition, scored -2.05 against an opening of 0.00.
    #
    # The ladder is the evidence the change is right rather than merely different: before
    # it, `minimax:nodes=6400` rated level with a random mover and the 400-node search
    # finished last (`ladder_monotonicity` 0.33). After it the ladder is monotonic --
    # random -686, greedy -225, then 89, 303, 519.
    "antichess": "73fde0b76c57dee4",
    "horde": "a17dbbca70b40efa",
    "threecheck": "ac39bd9e60fe80e6",
}

# `HAND` is the third, and the same shape of promise as `PHASE_B`: a term that reads a
# rule moves exactly the rulesets that declare it. `v11` A6 (Q389, 2026-09-24) made the
# hand material, and the hand exists only under `drops`, so of `INERT` exactly the two
# that declare `drops` moved -- crazyhouse from `ce435766743f7917` and dobutsu from
# `25b2f861b2b206c7` -- and the other eleven held to the digit. They were in `INERT`
# because no *goal* term reads a hand; the table's claim is about evaluation terms, and
# the hand is one now.
HAND = {
    "crazyhouse": "b92c261e6719fb63",
    "dobutsu": "d5b311ca12a00a4e",
}

# Spelled `threats=0`: these tables pin what *evaluation* terms move, and Q502i's
# threat-aware quiescence is a search change, on by default since, that moved four of
# them (threecheck, shatranj, xiangqi, crazyhouse) without any term reading anything.
# Pinning the search they were written against keeps each table's claim about terms.
_BEFORE_Q502I = "threats=0"
_PAIRINGS = [
    ("greedy", f"minimax:nodes=400,{_BEFORE_Q502I}"),
    (f"minimax:nodes=800,{_BEFORE_Q502I}", f"minimax:nodes=800,{_BEFORE_Q502I}"),
]
_SEEDS = (0,)


def _digest(name: str) -> str:
    engine = _engine(name)
    sha = hashlib.sha256()
    for agents in _PAIRINGS:
        for seed in _SEEDS:
            record = play_game(engine, agents, seed)
            sha.update(repr(record.moves).encode())
            sha.update(f"{record.winner}|{record.reason}|{record.nodes}".encode())
            sha.update(repr(record.material).encode())
            sha.update(repr(record.branching).encode())
    return sha.hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(INERT))
def test_a_ruleset_that_declares_no_goal_rule_plays_the_game_it_always_did(name: str) -> None:
    """Same move lists, same node counts, same recorded series, to the digit.

    A term accidentally added to the ungated path is exactly what this catches, and it
    caught three of them: a terminal check hoisted for every ruleset rather than for the
    ones whose endings the evaluator reads; when that was gated, a search that then asked
    only the half of the verdict that needs a move list, and so could no longer see a
    repetition, a clock or a royal capture; and, when phase B widened the gate, the six
    rulesets below and no others.
    """
    assert _digest(name) == INERT[name]


@pytest.mark.parametrize("name", sorted(PHASE_B))
def test_a_ruleset_phase_b_reads_plays_the_game_phase_b_gave_it(name: str) -> None:
    """The other side of the same assertion.

    These six were inert through phase A and moved in phase B, which is the whole content
    of "the blast radius is the set of rules the term reads". A phase-C change to the
    search would move both tables; a further evaluation term should move only this one.
    """
    assert _digest(name) == PHASE_B[name]


@pytest.mark.parametrize("name", sorted(HAND))
def test_a_ruleset_with_a_hand_plays_the_game_the_hand_term_gave_it(name: str) -> None:
    """The rulesets A6's hand term reads, pinned on the game it gave them.

    Moved together and alone: `INERT`'s other eleven, none of which declares `drops`,
    did not move when this table was written.
    """
    assert _digest(name) == HAND[name]


# ------------------------------------------------------------------ E5: flat roots


def _flatness(name: str, plies: int = 40) -> tuple[float, float]:
    """The share of sampled positions where every legal move scores the same, and the
    mean share of the move list tied for best.

    Sampled off a self-play game rather than a fixed position list, which is the honest
    way to ask the question and has one consequence worth stating: a ruleset whose
    *values* changed plays a different game, so its positions are different ones. Two
    probes moved for exactly that reason and neither is asserted on below -- see the
    table in ``docs/reference/stage-17-measurements.md``.
    """
    engine = _engine(name)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    record = play_game(engine, ("minimax:nodes=1600",) * 2, 0, collect_series=False)

    state = engine.initial_state()
    flat = tied = positions = 0.0
    for index, move in enumerate(record.moves[:plies]):
        if index % 2 == 0:
            legal = engine.legal_moves(state)
            if len(legal) > 1:
                scores = _score_roots(engine, evaluator, state, legal)
                best = max(scores)
                n_best = sum(1 for score in scores if score == best)
                flat += n_best == len(scores)
                tied += n_best / len(scores)
                positions += 1
        state = engine.apply(state, move)
    return (flat / positions, tied / positions) if positions else (0.0, 0.0)


@pytest.mark.parametrize(
    "name,tied_before", [("rps-v2", 0.6149005506898694), ("rps-v4", 0.6149005506898694)]
)
def test_the_probe_can_tell_the_moves_apart_on_a_game_won_by_ground(
    name: str, tied_before: float
) -> None:
    """`decision_criticality` runs the evaluator, so on a game whose costs the evaluator
    cannot price it measures the evaluator.

    Before stage 17A the probe could not separate *any* legal move in 10% of these
    positions and rated 61% of the move list tied for best, against chess's 5% and 22%.
    The component's own docstring says it measures "the share of the legal move list that
    would actually cost something", and on these rulesets it was measuring nothing of the
    kind. It is scored, at 0.099, with the best discrimination evidence in the set.
    """
    fully_flat, tied = _flatness(name)
    assert fully_flat == 0.0
    assert tied < tied_before - 0.1


def test_the_control_moves_only_when_the_search_does() -> None:
    """Chess is the control, and what it controls for is worth being precise about.

    Through stage 17A and 17B it did not move at all -- 0.05 fully flat and 0.2244 tied,
    exactly what the build before stage 17A measured -- which is what said the *probe*
    had not got easier while the goal terms were being added. Phase C changed the search,
    so chess plays different games, so the probe is asked about different positions and
    the number moves. That is not the failure this test guards against; it is the
    difference between "the instrument changed" and "the games changed", and the literal
    is re-pinned rather than relaxed so the next evaluation term still has to face it.
    """
    assert _flatness("chess") == (0.05, 0.09578391815227519)


def test_a_game_won_by_capture_alone_is_left_flat() -> None:
    """`rps-v1` is the honest negative in this set, and it barely improves.

    It is won by annihilation, which is a capture ending -- material already reads it --
    so no goal term applies, and its army is two of each piece, so every piece may take
    the same share of the same enemy and the relational values are all equal too. Its
    flatness is a symmetric opening in which every move really is equivalent, which is
    the reason its designer rejected it and not a reason to add a term. 0.80 fully flat
    before, 0.65 after. Recorded rather than tuned away: re-aiming a measurement after
    seeing it miss is fitting the instrument to one designer's taste.
    """
    fully_flat, _ = _flatness("rps-v1")
    chess_flat, _ = _flatness("chess")
    # A ratio rather than a threshold. The absolute number has come down as the search
    # got deeper -- 0.80 before stage 17A, 0.65 after it, 0.45 after phase C -- because a
    # deeper search visits different positions, and pinning it would be pinning the
    # search rather than the property. What does not move is the gap: rps-v1's opening is
    # symmetric in a way chess's is not, and the probe finds nothing to separate there
    # roughly an order of magnitude more often.
    assert fully_flat > 5 * chess_flat, f"{fully_flat} against chess's {chess_flat}"
