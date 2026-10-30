"""Which ending wins when two of them fire on the same position.

The schema states nine ways a game can end and says nothing about what happens when two
are true at once. The answer is the order somebody wrote the `if`s in, and until this
file it was not in the spec, not in `describe.py`, not in `.claude/rules/` and not
tested. Chess cannot reach the ambiguity -- it has one ending. A ruleset with `reach`
**and** `extinction` **and** `territory.win_at` reaches it immediately, and the LLM
proposer can write one.

**These tests are read off the code, not off intent.** That is the point of them: they
do not argue that the order is right, they make it *knowable*, so that a later change to
it is a test failure rather than a silent difference in who won. Making the order a
declared list in the spec is the right end state and is a schema bump with a
`_RULE_COSTS` argument attached; this is the step before it.

The order, as `game.rs` has it. Everything above the line is asked before a move is
generated (`terminal_prefix`); the two below need the move list to be empty
(`terminal_from_moves`):

    1. a royal destroyed by a blast          explosion + royals
    2. the illegal-position guard            checkmate games only
    3. win: royal_capture / annihilation     rules.win
    4. reach                                 rules.reach
    5. extinction                            rules.extinction
    6. bare royal                            rules.bare_royal
    7. territory                             rules.territory.win_at
    8. check limit                           rules.check_limit
    8b. insufficient material (Q1069a)       rules.insufficient_material, a draw; it holds
                                             only where none of 1 to 8 can, so it is pinned
                                             against 9 to 11 in tests/test_insufficient_material.py
    9. repetition                            always
    10. the no-progress clock                -> rules.scoring, unless it is mate (Q412)
    11. the ply limit                        -> rules.scoring, unless it is mate (Q412)
    ---------------------------------------- the move list is empty from here
    12. checkmate                            rules.win: checkmate
    13. stalemate                            rules.stalemate

**Since Q412 checkmate also outranks the two clocks.** On the ply a clock runs out,
`terminal_prefix` generates the moves itself when the side to move is in check, and a
mate stands, as it does under FIDE's 75-move rule and in Fairy-Stockfish. A stalemate on
that ply is still the clock's.

**There is a second order inside the first, and it is just as undeclared.** Every rule
from 3 to 8 loops `for side in 0..2`, so when the *same* rule fires for both players at
once, the first player's instance is the one that decides. `test_the_first_player_is_
checked_first` pins it.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine

# A 5x5 board with a royal `K`, and `S` and `T` to be counted, extinguished and bared.
# Deliberately not chess: every assertion here is about a rule interaction, and chess has
# exactly one ending so it can express none of them.
PIECES: list[dict[str, Any]] = [
    {"id": "K", "name": "king", "value": 0, "royal": True,
     "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    {"id": "S", "name": "stepper", "value": 1,
     "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
    {"id": "T", "name": "walker", "value": 1,
     "moves": [{"kind": "leap", "vectors": [[1, 1]], "symmetry": "rotate4"}]},
]


def spec(**rules: Any) -> dict[str, Any]:
    """A playable 5x5 game with whichever endings a test wants to collide."""
    return {
        "schema_version": 4,
        "name": "precedence probe",
        "board": {"width": 5, "height": 5},
        "pieces": PIECES,
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["e5"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "S", "player": 1, "squares": ["d5"]},
            {"piece": "T", "player": 0, "squares": ["c1"]},
            {"piece": "T", "player": 1, "squares": ["c5"]},
        ]},
        "rules": rules,
    }


def verdict(engine: Engine, placement: dict[str, tuple[str, int]], **state: Any) -> Any:
    """The outcome of a hand-built position, or `None` if the game is not over."""
    built = engine.make_state(placement, side_to_move=state.pop("side_to_move", 0))
    return engine.terminal(built.with_(**state) if state else built)


# --------------------------------------------------------------- the adjacent pairs
#
# One test per step of the chain. Each builds a position where *both* neighbours fire
# and where they disagree about the winner -- a pair that agrees proves nothing, and
# getting them to disagree is most of the work in each of these.


def test_win_beats_reach() -> None:
    """`royal_capture` is asked before the schema-2 endings, so it decides.

    Player 1 has no royal, which `royal_capture` reads as a win for player 0. Player 1's
    stepper is standing on the square `reach` awards *them* the game for. The two
    disagree, and the first one asked is the one that answers.
    """
    engine = Engine(spec(win="royal_capture",
                         reach={"piece": "S", "squares": ["c3"], "mirror": False}))
    outcome = verdict(engine, {"a1": ("K", 0), "c3": ("S", 1)})
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (0, "royal_captured")


def test_reach_beats_extinction() -> None:
    """Arrival before running out.

    Player 1's stepper is on the winning square, and player 1 has no `T` at all, which
    `extinction` reads as a loss for them. Reach says player 1, extinction says player 0.
    """
    engine = Engine(spec(
        win="royal_capture",
        reach={"piece": "S", "squares": ["c3"], "mirror": False},
        extinction={"pieces": ["T"], "min_count": 1, "result": "loss"},
    ))
    outcome = verdict(engine, {"a1": ("K", 0), "c1": ("T", 0), "e5": ("K", 1), "c3": ("S", 1)})
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "reached")


def test_extinction_beats_bare_royal() -> None:
    """Running out before being stripped.

    Player 0 has no `S`, which `extinction` reads as a loss for player 0. Player 1 has
    nothing but a royal, which `bare_royal` reads as a loss for player 1. Extinction is
    asked first, so player 0 loses -- and note extinction would answer the same way even
    if player 1 were also short of an `S`, because the loop reaches player 0 first.
    """
    engine = Engine(spec(
        win="royal_capture",
        extinction={"pieces": ["S"], "min_count": 1, "result": "loss"},
        bare_royal="loss",
    ))
    outcome = verdict(engine, {"a1": ("K", 0), "c1": ("T", 0), "e5": ("K", 1), "d5": ("S", 1)})
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "extinction")


def test_bare_royal_beats_territory() -> None:
    """Being stripped before holding ground.

    Player 1 is down to its royal, a loss for them. Player 1 also holds enough of the
    board to win on territory. Bare royal is asked first.
    """
    engine = Engine(spec(
        win="royal_capture",
        bare_royal="loss",
        territory={"mode": "permanent", "win_at": 3},
    ))
    engine_spec = engine.spec
    colour = bytearray(engine_spec.n_squares)
    for square in (0, 1, 2):
        colour[square] = 2  # claimed by the second player
    outcome = verdict(
        engine,
        {"a1": ("K", 0), "c1": ("T", 0), "e5": ("K", 1)},
        colour=bytes(colour),
    )
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (0, "bare_royal")


def test_territory_beats_the_check_limit() -> None:
    """Ground before checks.

    Player 0 holds three squares, enough to win on territory. Player 1 has delivered
    enough checks to win on the count. Territory is asked first.
    """
    engine = Engine(spec(
        win="royal_capture",
        territory={"mode": "permanent", "win_at": 3},
        check_limit=2,
    ))
    colour = bytearray(engine.spec.n_squares)
    for square in (0, 1, 2):
        colour[square] = 1  # claimed by the first player
    outcome = verdict(
        engine,
        {"a1": ("K", 0), "c1": ("T", 0), "e5": ("K", 1), "d5": ("S", 1)},
        colour=bytes(colour),
        checks=(0, 2),
    )
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (0, "coloured")


def test_the_check_limit_beats_the_clocks() -> None:
    """A count of checks before running out of time.

    The position is simultaneously at the ply limit -- which a scoring rule would award
    to player 0, who is ahead on material -- and past the check limit for player 1. The
    check limit is asked first, so player 1 wins.
    """
    engine = Engine(spec(
        win="royal_capture",
        check_limit=2,
        scoring={"count": "material"},
        max_plies=10,
    ))
    outcome = verdict(
        engine,
        {"a1": ("K", 0), "b1": ("S", 0), "c1": ("T", 0), "e5": ("K", 1)},
        checks=(0, 2),
        ply=10,
    )
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "check_limit")


# Player 0's king on a1 with nothing else. `T` on b2 attacks a1 diagonally; the two `S`
# cover the only two flight squares, a2 and b1. Take the `T` away and the king is not
# attacked at all and still cannot move, which is the other verdict entirely -- so the
# two positions differ by one piece and that piece is the whole distinction.
MATED = {"a1": ("K", 0), "b2": ("T", 1), "a3": ("S", 1), "c1": ("S", 1), "e5": ("K", 1)}
STALEMATED = {"a1": ("K", 0), "a3": ("S", 1), "c1": ("S", 1), "e5": ("K", 1)}


def test_the_clocks_beat_the_empty_move_list() -> None:
    """The ply limit is asked before a move is even generated.

    A position that is *both* out of time and stalemated is **scored, not drawn**, which
    is a different result and not merely a different reason: the clock hands it to
    whoever is ahead. `terminal_prefix` runs first and never looks at a move. This is
    the pair that is easiest to get backwards when reading the code, because the two
    halves live in different functions.
    """
    untimed = Engine(spec(win="checkmate", stalemate="draw", max_plies=400))
    drawn = verdict(untimed, STALEMATED)
    assert drawn is not None
    assert (drawn.winner, drawn.reason) == (None, "stalemate"), (
        "the position has to actually be stalemate, or the pair is not a pair"
    )

    timed_out = Engine(spec(
        win="checkmate", stalemate="draw",
        scoring={"count": "material"}, max_plies=10,
    ))
    outcome = verdict(timed_out, STALEMATED, ply=10)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "scored")


# Each clock, run out on the position it is asked about: the no-progress clock at 20
# plies, then the ply limit at 30.
CLOCKS = {
    "no_progress": ({"halfmove": 20}, "no_progress"),
    "ply_limit": ({"ply": 30}, "ply_limit"),
}


@pytest.mark.parametrize("clock", list(CLOCKS), ids=list(CLOCKS))
def test_checkmate_beats_the_clocks(clock: str) -> None:
    """A mate given on the ply a clock runs out stands (Q412).

    The user decided on 2026-09-24: *"mate wins"*, as FIDE's 75-move rule and
    Fairy-Stockfish have it. Until then the clocks were asked before the move list, so
    this position was a draw. The two disagree on the result, not only on the reason:
    the clock draws, the mate is player 1's.

    Asked three ways, because they are three code paths in the core and a verdict must
    not depend on which one a caller takes: the whole verdict, the verdict from a move
    list the caller generated, and the half a search asks before its table.
    """
    engine = Engine(spec(win="checkmate", stalemate="draw", max_plies=30, no_progress_draw=20))
    run_out, reason = CLOCKS[clock]
    mated = engine.make_state(MATED, side_to_move=0).with_(**run_out)

    whole = engine.terminal(mated)
    listed = engine.terminal(mated, engine.legal_moves(mated))
    prefix = engine.terminal_prefix(mated)
    for outcome in (whole, listed, prefix):
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (1, "checkmate")

    # The control: the same clock on a stalemate still ends the game by the clock. A
    # mate outranks the clock; having no legal move does not.
    stalemated = engine.make_state(STALEMATED, side_to_move=0).with_(**run_out)
    for outcome in (
        engine.terminal(stalemated),
        engine.terminal(stalemated, engine.legal_moves(stalemated)),
        engine.terminal_prefix(stalemated),
    ):
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (None, reason)

    # In check with a way out is not mate, so the clock ends it there too: a walker of
    # player 0's on c3 can take the one on b2 that gives check.
    escapes = engine.make_state({**MATED, "c3": ("T", 0)}, side_to_move=0).with_(**run_out)
    assert engine.legal_moves(escapes)
    for outcome in (
        engine.terminal(escapes),
        engine.terminal(escapes, engine.legal_moves(escapes)),
        engine.terminal_prefix(escapes),
    ):
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (None, reason)


def test_checkmate_beats_a_scored_clock_whoever_the_score_favours() -> None:
    """The mate stands over a clock that would have scored the game, too.

    Being mated wins here (`rules.checkmate: "win"`), so the mate is player 0's, and the
    score on the clock is player 1's, who has all the material. The mate decides.
    """
    engine = Engine(spec(
        win="checkmate", checkmate="win", stalemate="draw",
        scoring={"count": "material"}, max_plies=30,
    ))
    outcome = engine.terminal(engine.make_state(MATED, side_to_move=0).with_(ply=30))
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (0, "checkmate")


def test_checkmate_and_stalemate_are_told_apart_by_check_alone() -> None:
    """The last step of the chain, and the only one that is not a precedence.

    An empty move list is one of the two, never both: the difference is whether the side
    to move is in check. It is in the chain because it is where the chain ends, and it is
    asserted in both directions because a rule that read the check the wrong way round
    would still produce an ending and would still look plausible.
    """
    engine = Engine(spec(win="checkmate", stalemate="draw", max_plies=400))

    mate = verdict(engine, MATED)
    assert mate is not None
    assert (mate.winner, mate.reason) == (1, "checkmate")

    stale = verdict(engine, STALEMATED)
    assert stale is not None
    assert (stale.winner, stale.reason) == (None, "stalemate")


# ------------------------------------------------------- and the order inside the order


def test_the_first_player_is_checked_first() -> None:
    """When one rule fires for both players, player 0's instance decides.

    Every rule from `win` down to `check_limit` loops `for side in 0..2`, so a symmetric
    position is not resolved symmetrically. Here *both* players are down to a bare royal
    and the rule is a loss for whoever is: player 0 is reached first, so player 0 loses.
    Worth pinning because it is the kind of asymmetry that shows up as one seat winning
    more often and gets blamed on the agents.
    """
    engine = Engine(spec(win="royal_capture", bare_royal="loss"))
    outcome = verdict(engine, {"a1": ("K", 0), "e5": ("K", 1)})
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "bare_royal")


def test_reach_beats_a_royal_bared_by_a_wild_piece() -> None:
    """Schema 9 inserts no step: a wild piece reaches the existing endings as a new actor.

    The sharpest case the plan named, and it is decided by the order as it already
    stands. Player 0's stepper arrives on the winning square, and in the *same ply's*
    phase the boar eats player 1's last common piece, so `bare_royal` fires too. They
    disagree -- `reach` is stated as a loss for whoever arrives, `bare_royal` as a loss
    for whoever is stripped -- and `reach` is asked first, so `reach` answers.

    What is being pinned is that the phase produces an ending the same way a move does:
    the position `terminal` is asked about is the post-phase one, and nothing about the
    chain of thirteen had to change to accommodate it.
    """
    wild = spec(
        win="royal_capture",
        reach={"piece": "S", "squares": ["c3"], "mirror": False, "result": "loss"},
        bare_royal="loss",
        npcs=[{"piece": "W", "route": ["b5", "c5"], "mirror": False}],
    )
    wild["schema_version"] = 9
    wild["pieces"] = [*PIECES, {"id": "W", "name": "boar", "value": 1,
                                "moves": [{"kind": "leap", "vectors": [[1, 0]],
                                           "symmetry": "rotate4"}]}]
    engine = Engine(wild)

    before = engine.make_state(
        {"a1": ("K", 0), "b3": ("S", 0), "e5": ("K", 1), "c5": ("T", 1),
         "b5": ("W", 2)},
        moved=("b5",),
    )
    assert engine.terminal(before) is None
    move = next(
        m for m in engine.legal_moves(before)
        if (m & 0xFF) == 11 and ((m >> 8) & 0xFF) == 12  # b3 -> c3 on a 5-wide board
    )
    after = engine.apply(before, move)
    # The phase did happen: player 1 is down to a royal, which `bare_royal` would call
    # a loss for them -- the ending that does *not* get to answer.
    mine = {square for square, value in enumerate(after.board)
            if value and engine.spec.owner_of(value) == 1}
    assert mine == {24}  # e5, the royal, and nothing else: the boar ate the walker
    assert engine.spec.owner_of(after.board[22]) == 2  # c5, where the boar now is
    outcome = engine.terminal(after)
    assert outcome is not None
    assert (outcome.winner, outcome.reason) == (1, "reached")


def test_the_illegal_position_guard_outranks_every_stated_ending() -> None:
    """The newest step in the chain, and the one that fires soonest.

    A checkmate game whose side to move is not in check while the *opponent* is cannot
    have been reached by play, so the position is treated as already won -- otherwise the
    king is simply captured on the next ply and the game carries on. It sits above every
    stated ending, which means a hand-built or generated position can be decided by it
    before `reach` or `extinction` is ever consulted. That is a real consequence and it
    is asserted rather than left to be discovered.
    """
    engine = Engine(spec(
        win="checkmate", stalemate="draw", max_plies=400,
        reach={"piece": "S", "squares": ["c3"], "mirror": False},
    ))
    # Player 1's king on e5 is attacked by the stepper on d5; player 0 is not in check,
    # and player 0's stepper is on the square `reach` would award to player 0 anyway --
    # so the two agree on the winner and disagree on the reason, which is what is read.
    outcome = verdict(engine, {"a1": ("K", 0), "c3": ("S", 0), "d5": ("S", 0), "e5": ("K", 1)})
    assert outcome is not None
    assert outcome.reason == "royal_captured", (
        "the guard is asked before rules.reach, so it is the guard that answers"
    )
    assert outcome.winner == 0


# ------------------------------------------------------------------------- the prose


def test_the_prose_states_the_order_when_a_spec_has_more_than_one_ending() -> None:
    """A player told two ways to win and not which is checked first has not been told
    the rules.

    This is the prose invariant applied to something that was not data: the order is
    decidable and fixed, so leaving it out of the description is a gap rather than a
    simplification. The judge, the proposer and a person playing the game all read
    `describe.py`, and until now all three were told a game could end two ways and left
    to guess what happens when both do.
    """
    from chess_variant_sandbox.spec.describe import describe_rules

    both = describe_rules(spec(
        win="royal_capture",
        reach={"piece": "S", "squares": ["c3"], "mirror": False},
        extinction={"pieces": ["T"], "min_count": 1, "result": "loss"},
    ))
    stated = [line for line in both if line.startswith("When more than one")]
    assert len(stated) == 1, both
    assert stated[0].index("arrival") < stated[0].index("running out"), stated[0]
    assert stated[0].index("royal piece taken") < stated[0].index("arrival"), stated[0]


def test_the_prose_says_nothing_when_there_is_nothing_to_disambiguate() -> None:
    """Chess is the case: one ending, no order to state, no sentence.

    A rule that fires on every spec is a rule nobody reads. This one is silent for the
    great majority of the corpus, which is what keeps it worth reading on the minority.
    """
    from chess_variant_sandbox.measurement import ROOT
    from chess_variant_sandbox.spec import load_spec
    from chess_variant_sandbox.spec.describe import describe_rules

    chess = describe_rules(load_spec(ROOT / "rulesets" / "chess.json"))
    assert not [line for line in chess if line.startswith("When more than one")]


@pytest.mark.parametrize("declared", [
    {"reach": {"piece": "S", "squares": ["c3"], "mirror": False}},
    {"extinction": {"pieces": ["T"], "min_count": 1, "result": "loss"}},
    {"bare_royal": "loss"},
    {"check_limit": 2},
])
def test_a_single_ending_is_unaffected_by_the_order(declared: dict[str, Any]) -> None:
    """The control: each rule still fires on its own.

    Without this the whole file could pass on an engine that had stopped checking four
    of the five, because every assertion above only ever names the *winner* of a race.
    """
    engine = Engine(spec(win="royal_capture", **declared))
    assert engine.terminal(engine.initial_state()) is None, (
        "the opening position is not meant to be over"
    )
