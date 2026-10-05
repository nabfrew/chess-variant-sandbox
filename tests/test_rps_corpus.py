"""The rock-paper-scissors corpus: schema 3's first real exercise, and a labelled set.

These are not synthetic test specs. They are games somebody designed, played, rejected,
and said why about -- written up in `docs/reference/rps-versions.md` with the verdicts
attached. That makes them the one thing `metrics/mutants.py` can never be: a corpus with
**independent labels**, produced by playtesting, by someone who has never seen this index.

Six of the seven versions are buildable, and each one names the feature it needed:

* `rps-hierarchy-chess`   the precursor, committed unfixed with its immortal king
* `rps-v1`                the annihilation game, three pieces a side (schema 3)
* `rps-v3`                the race, a larger army and a win by arrival (schema 3)
* `rps-v2`                permanent territory, first to 41 squares (schema 4)
* `rps-v4`                the same board, repaintable (schema 4)
* `rps-v5`                both endings at once, ground compared at the clock (schema 4)

Version 6 is the one still missing, and deliberately: it needs a friendly-screen hop --
which the engine has -- and an army size the source never states, so committing a guess
as a labelled negative would be inventing data.

They are **probes, never band members**. All three are negatives, so the set can show
that a named failure is *found* and can never show that a good game is *accepted* --
and fitting curves until the index agreed with one designer's taste would repeat the
mistake the mutant corpus already caught, where four "obviously broken" labels turned
out to be wrong and the index was right every time.

The perft literals below are **this engine's own**, unlike `test_variants.py`, where
every number is Fairy-Stockfish's. Nothing else implements a cyclic capture relation, so
there is nothing to check against; they are frozen as regression pins, and a number that
moves is a movegen regression rather than a new behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.spec.schema import validate_spec

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
CORPUS = (
    "rps-v1", "rps-v3", "rps-hierarchy-chess", "rps-v2", "rps-v4", "rps-v5",
)
FAST_BUDGET_NODES = 150_000

OPENING_PERFT: dict[str, dict[int, int]] = {
    "rps-v1": {1: 15, 2: 225, 3: 3_540, 4: 55_696},
    "rps-v3": {1: 35, 2: 1_225, 3: 44_065, 4: 1_585_081},
    "rps-hierarchy-chess": {1: 32, 2: 944, 3: 37_796, 4: 1_447_016},
    # The three territory versions share v3's army and movement, so they share its move
    # counts exactly -- colouring a square changes no move, only what the position is
    # worth. That equality is worth having as a literal: it is the cheapest statement
    # that territory is inert on move generation.
    "rps-v2": {1: 35, 2: 1_225, 3: 44_065},
    "rps-v4": {1: 35, 2: 1_225, 3: 44_065},
    "rps-v5": {1: 35, 2: 1_225, 3: 44_065},
}

_ENGINES: dict[str, Engine] = {}


def engine_for(name: str) -> Engine:
    if name not in _ENGINES:
        _ENGINES[name] = Engine(load_spec(RULESETS / f"{name}.json"))
    return _ENGINES[name]


def perft_cases() -> list[pytest.ParameterSet]:
    cases = []
    for name, ladder in OPENING_PERFT.items():
        for depth, expected in sorted(ladder.items()):
            marks = [pytest.mark.slow] if expected > FAST_BUDGET_NODES else []
            cases.append(pytest.param(name, depth, expected, id=f"{name}-d{depth}", marks=marks))
    return cases


@pytest.mark.parametrize(("name", "depth", "expected"), perft_cases())
def test_opening_perft_is_frozen(name: str, depth: int, expected: int) -> None:
    engine = engine_for(name)
    assert engine.perft(engine.initial_state(), depth, stop_at_terminal=True) == expected


@pytest.mark.parametrize("name", CORPUS)
def test_every_ruleset_is_valid_and_declares_the_version_it_needs(name: str) -> None:
    """The declared version is a claim about which features the rules use, and the
    validator enforces it in both directions -- so this only has to check the claim is
    the *smallest* one that works."""
    spec = load_spec(RULESETS / f"{name}.json")
    assert validate_spec(spec) == []
    assert any("captures" in piece for piece in spec["pieces"]), "all six use the matrix"

    needs_territory = "territory" in spec["rules"]
    assert spec["schema_version"] == (4 if needs_territory else 3)

    # ...and one lower is refused, by name.
    lowered = dict(spec, schema_version=spec["schema_version"] - 1)
    assert any("schema" in error for error in validate_spec(lowered))


class TestTheCaptureCycle:
    """What the DSL could not state before, checked as a relation rather than a table.

    Every one of these is a claim from `docs/reference/rps-versions.md`'s common core:
    "rock takes scissors, scissors takes paper, paper takes rock. No piece may capture
    its own type, and none may capture the type that beats it."
    """

    def test_each_piece_takes_exactly_one_type_and_is_taken_by_exactly_one(self) -> None:
        from chess_variant_sandbox.engine.compile import compile_spec

        matrix = compile_spec(load_spec(RULESETS / "rps-v1.json")).capture_allowed
        for attacker, row in enumerate(matrix):
            assert sum(row) == 1, f"piece {attacker} preys on {sum(row)} types"
            assert not row[attacker], "no piece may capture its own type"
        for victim in range(len(matrix)):
            takers = sum(1 for row in matrix if row[victim])
            assert takers == 1, f"piece {victim} is prey to {takers} types"
        # ...and the relation is a cycle rather than a hierarchy: nothing is safe from
        # everything, and nothing preys on everything. A value order would fail this.
        assert all(not matrix[a][b] or not matrix[b][a] for a in range(3) for b in range(3))

    def test_the_generator_offers_only_the_capture_the_cycle_permits(self) -> None:
        from chess_variant_sandbox.engine.moves import move_from, move_to

        engine = engine_for("rps-v1")
        square = engine._square_index
        # A rock with all three enemy types adjacent to it.
        state = engine.make_state(
            {
                "e5": ("R", 0),
                "e6": ("S", 1),  # prey
                "d5": ("R", 1),  # its own type: no
                "f5": ("P", 1),  # the type that takes it: no
            },
            side_to_move=0,
        )
        targets = {
            move_to(m) for m in engine.legal_moves(state) if move_from(m) == square("e5")
        }
        assert square("e6") in targets, "rock takes scissors"
        assert square("d5") not in targets, "rock may not take rock"
        assert square("f5") not in targets, "rock may not take the paper that takes it"
        assert square("e4") in targets, "and it may still step onto an empty square"

    def test_a_piece_it_may_not_capture_blocks_a_line_rather_than_being_taken(self) -> None:
        """Hierarchy chess is all rooks, so this is where the ray case shows up.

        An occupied square ends a slide whether or not the piece can be taken. A
        capture-matrix implementation that `continue`d instead of `break`ing would let a
        rook reach straight through a piece it was not allowed to capture, which perft
        would count and no assertion here would otherwise see.
        """
        from chess_variant_sandbox.engine.moves import move_from, move_to

        engine = engine_for("rps-hierarchy-chess")
        square = engine._square_index
        # A pawn may capture nothing at all (nothing is below it in the order). A queen
        # stands two squares away with an enemy pawn between them.
        state = engine.make_state(
            {"a1": ("R", 0), "a3": ("K", 1), "a5": ("P", 1), "h8": ("K", 0)},
            side_to_move=0,
        )
        targets = {
            move_to(m) for m in engine.legal_moves(state) if move_from(m) == square("a1")
        }
        assert square("a2") in targets, "the empty square before the blocker is reachable"
        assert square("a3") not in targets, "a rook may not capture a king"
        assert square("a4") not in targets, "and it may not slide through one either"
        assert square("a5") not in targets


class TestVersionOnesForcedDraw:
    """The stated game-theoretic property of version 1, checked rather than assumed.

    "From a balanced position a player can sacrifice a paper; whichever way the opponent
    answers, the game reduces to both sides holding only a rock. Neither rock can capture
    the other, so the position is dead drawn."

    This is worth a test of its own because it exercises the capture matrix and terminal
    detection *together*, in a way no perft can: a perft counts moves and would happily
    count them forever in a position where nothing can ever be resolved.
    """

    def rock_endgame(self) -> tuple[Engine, object]:
        engine = engine_for("rps-v1")
        return engine, engine.make_state({"c3": ("R", 0), "g7": ("R", 1)}, side_to_move=0)

    def test_neither_rock_can_touch_the_other(self) -> None:
        from chess_variant_sandbox.engine.moves import move_to

        engine, state = self.rock_endgame()
        enemy = engine._square_index("g7")
        # Walk the rock right up to the enemy and check the square stays unavailable.
        for square_name in ("f6", "f7", "g6", "h8"):
            adjacent = engine.make_state(
                {square_name: ("R", 0), "g7": ("R", 1)}, side_to_move=0
            )
            assert enemy not in {move_to(m) for m in engine.legal_moves(adjacent)}, (
                f"a rock on {square_name} was offered the enemy rock"
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("nodes", [400, 6400])
    def test_no_search_at_any_budget_finds_a_win(self, nodes: int) -> None:
        """Played out, not merely inspected. The claim is that the position is dead."""
        from chess_variant_sandbox.tournament.play import finish_game

        engine, state = self.rock_endgame()
        agent = f"minimax:nodes={nodes}"
        record = finish_game(engine, (agent, agent), state, seed=0)
        assert record.winner is None, (
            f"a {nodes}-node search claimed a win in a position with no legal capture "
            f"in it, ending by {record.reason}"
        )

    def test_the_position_ends_and_ends_drawn(self) -> None:
        """It is a draw by exhaustion rather than a draw by rule, and that is the point:
        the game has no way to make progress, so its own clocks end it."""
        from chess_variant_sandbox.tournament.play import finish_game

        engine, state = self.rock_endgame()
        record = finish_game(engine, ("greedy", "greedy"), state, seed=0)
        assert record.winner is None
        assert record.reason in ("no_progress", "ply_limit", "repetition")


class TestTheGatesReadTheseCorrectly:
    """A free validation of two existing gates, which is half of why these are committed.

    `docs/reference/rps-versions.md` predicted both of these before the rulesets existed,
    and the point of writing a prediction down first is that it cannot be fitted after.
    """

    @pytest.mark.slow
    def test_an_unwinnable_ruleset_is_named_as_unwinnable(self) -> None:
        """Hierarchy chess as its designer first stated it: nothing outranks the king,
        so the king cannot be captured, so the game never ends.

        The prediction was that `gates.py`'s "none of N random games produced a winner"
        would catch it. It does, by name.
        """
        from chess_variant_sandbox.loop import smoke_test

        result = smoke_test(load_spec(RULESETS / "rps-hierarchy-chess.json"), games=60)
        assert not result.passed
        assert any("may be out of reach" in reason for reason in result.reasons), result.reasons

    @pytest.mark.slow
    def test_version_ones_draws_are_visible_to_the_cheap_gate(self) -> None:
        """The designer's second verdict on version 1: "no one will be able to force a
        win". The smoke gate reaches the same conclusion from random play alone.

        Recorded as an agreement result, not as a threshold to tune. `MAX_RANDOM_STALL`
        is 0.98 and version 1 measures just above it, which is close -- but the gate was
        anchored on twenty-two committed rulesets before this game existed, and moving it
        now to change this answer would be fitting the instrument to the reading.

        **The rung above disagrees, and both are right.** `loop/screen.py` gives version
        1 four decisive games out of eight: a searching player wins it about half the
        time, and only random movers cannot. So the smoke gate stops this ruleset before
        the screen ever sees it, and the screen would have kept it. That is a fact about
        the ladder rather than about the game, it is exactly the shape of the shatranj
        mistake the gate was re-anchored to avoid, and stage 10 should look at it with
        the calibration corpus in hand rather than anyone adjusting a constant here.
        """
        from chess_variant_sandbox.loop import smoke_test

        result = smoke_test(load_spec(RULESETS / "rps-v1.json"), games=150)
        assert not result.passed
        assert any("ended on the turn limit or the no-progress rule" in reason
                   for reason in result.reasons), result.reasons

    @pytest.mark.slow
    def test_the_race_survives_the_gates(self) -> None:
        """Version 3 is a playable game with one stated flaw, and the flaw is not
        something random playouts can see -- it is about what happens after a mistake.
        A gate that rejected it would be rejecting a design for being subtle."""
        from chess_variant_sandbox.loop import smoke_test

        assert smoke_test(load_spec(RULESETS / "rps-v3.json"), games=150).passed


@pytest.mark.parametrize("name", CORPUS)
def test_the_evaluator_reads_a_restricted_army(name: str) -> None:
    """Piece values stop being mobility once capture is restricted.

    Every piece in version 1 moves identically, so a mobility-only evaluator scores all
    three the same -- and in hierarchy chess it scores a pawn that may capture nothing
    exactly as highly as a queen that may capture everything. The relational factor is
    part of stage 13 or stage 13 does not ship: a family of rulesets the roster cannot
    read is antichess's `ladder_monotonicity` problem by construction.
    """
    from chess_variant_sandbox.agents.evaluation import Evaluator

    engine = engine_for(name)
    values = Evaluator.for_engine(engine).values
    assert len(values) == engine.spec.n_types
    assert all(value >= 0 for value in values)

    if name in ("rps-v2", "rps-v4", "rps-v5"):
        return  # three pieces a side and no hierarchy; the cycle test covers them
    if name == "rps-hierarchy-chess":
        index = {pid: i for i, pid in enumerate(engine.spec.piece_ids)}
        # Same movement, different prey. A pawn preys on nothing and a queen on
        # everything below her, and a mobility-only evaluator could not tell them apart.
        assert values[index["P"]] < values[index["N"]] < values[index["Q"]]


@pytest.mark.parametrize("name", CORPUS)
def test_a_short_game_plays_through_with_the_hash_intact(name: str) -> None:
    """The check that perft cannot make. Three separate bugs have survived every perft
    in this suite because none of them was about counting moves."""
    import random

    engine = engine_for(name)
    rng = random.Random(11)
    state = engine.initial_state()
    for _ in range(30):
        moves = engine.legal_moves(state)
        if engine.terminal(state, moves) is not None:
            break
        state = engine.apply(state, rng.choice(moves))
        assert state.key == engine._compute_hash(state), f"{name} hash drifted"


class TestTerritory:
    """Squares that remember who moved onto them: schema 4, and versions 2, 4 and 5.

    The three share an army and a movement rule with version 3, so their perft ladders
    are identical to its -- colouring changes what a position is *worth*, never what
    moves exist. What is new is a per-square state that has to survive make, unmake,
    snapshot and the boundary, and a Zobrist argument that is different in each mode.
    """

    def test_moving_claims_the_square_and_permanent_means_permanent(self) -> None:
        engine = engine_for("rps-v2")
        square = engine._square_index
        state = engine.make_state({"e5": ("R", 0), "a1": ("R", 1)}, side_to_move=0)
        after = engine.apply(state, next(
            m for m in engine.legal_moves(state)
            if move_from(m) == square("e5") and move_to(m) == square("e6")
        ))
        assert after.colour[square("e6")] == 1, "the mover claims where it lands"
        assert after.colour[square("e5")] == 0, "and not where it came from"

        # The other player crossing it changes nothing, which is what permanent means.
        back = after.with_(side_to_move=1)
        moved = engine.apply(back, next(
            m for m in engine.legal_moves(back) if move_to(m) == square("a2")
        ))
        assert moved.colour[square("e6")] == 1

    def test_repainting_lets_the_other_player_take_it_back(self) -> None:
        engine = engine_for("rps-v4")
        square = engine._square_index
        # Paper for the second player, because rock cannot take rock: the piece that
        # comes to reclaim the square has to be allowed to capture what is standing on it.
        state = engine.make_state({"e5": ("R", 0), "e7": ("P", 1)}, side_to_move=0)
        first = engine.apply(state, next(
            m for m in engine.legal_moves(state) if move_to(m) == square("e6")
        ))
        assert first.colour[square("e6")] == 1
        second = engine.apply(first, next(
            m for m in engine.legal_moves(first) if move_to(m) == square("e6")
        ))
        assert second.colour[square("e6")] == 2, "a repainting square changes hands"

    def test_a_permanent_claim_is_irreversible_and_a_repaint_is_not(self) -> None:
        """The whole Zobrist argument, as a property rather than as a comment.

        A permanent claim can never be undone, so the ply that makes one resets the
        no-progress clock, so it clears the repetition window -- which is what lets the
        key leave colour out entirely. A repainting claim can be undone, so it does
        none of that and the key has to carry the colouring instead.
        """
        square = engine_for("rps-v2")._square_index
        for name, resets in (("rps-v2", True), ("rps-v4", False)):
            engine = engine_for(name)
            state = engine.make_state(
                {"e5": ("R", 0), "a1": ("R", 1)}, side_to_move=0, halfmove=5
            )
            after = engine.apply(state, next(
                m for m in engine.legal_moves(state) if move_to(m) == square("e6")
            ))
            assert (after.halfmove == 0) is resets, name

    def test_the_key_carries_a_repaint_and_not_a_permanent_claim(self) -> None:
        """Two positions identical but for the colouring.

        Under `repaint` they must hash differently or a repetition test would call them
        the same position. Under `permanent` they may hash the same, because the
        repetition window can never span a colour change -- and the test asserts that
        rather than leaving it as a claim.
        """
        square = engine_for("rps-v2")._square_index
        for name, differs in (("rps-v4", True), ("rps-v2", False)):
            engine = engine_for(name)
            plain = engine.make_state({"e5": ("R", 0), "a1": ("R", 1)}, side_to_move=0)
            painted = plain.with_(
                colour=bytes(
                    1 if index == square("c3") else 0
                    for index in range(engine.spec.n_squares)
                )
            )
            assert (
                engine._compute_hash(plain) != engine._compute_hash(painted)
            ) is differs, name

    @pytest.mark.parametrize("name", ["rps-v2", "rps-v4", "rps-v5"])
    def test_the_colouring_survives_a_hundred_plies_with_the_hash_intact(
        self, name: str
    ) -> None:
        """The check that catches what perft cannot.

        An explosion once cleared squares the incremental Zobrist update never saw, and
        the hash quietly disagreed with the position for the rest of the game. A new
        per-square state is the same shape of risk, so it is played rather than counted.
        """
        import random

        engine = engine_for(name)
        rng = random.Random(21)
        state = engine.initial_state()
        for _ in range(100):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            state = engine.apply(state, rng.choice(moves))
            assert state.key == engine._compute_hash(state), f"{name} hash drifted"
        assert any(state.colour), "something should have been claimed by now"

    def test_version_five_compares_ground_when_the_clock_runs_out(self) -> None:
        """Version 5's second ending, which is stage 14's scoring reading stage 16's
        state -- the two halves of the plan meeting."""
        engine = engine_for("rps-v5")
        square = engine._square_index
        board = {"e5": ("R", 0), "a1": ("R", 1)}
        colour = bytearray(engine.spec.n_squares)
        for name in ("b2", "c3", "d4"):
            colour[square(name)] = 1
        colour[square("h8")] = 2
        state = engine.make_state(board, side_to_move=0, ply=300).with_(
            colour=bytes(colour)
        )
        outcome = engine.terminal(state)
        assert outcome is not None
        assert outcome.reason == "scored"
        assert outcome.winner == 0, "three squares beat one"
