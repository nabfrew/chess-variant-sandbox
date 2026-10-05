"""The engine must be a rule interpreter, not a chess program with a config file.

Every ruleset here is deliberately un-chess-like: a different board shape, a movement
primitive chess does not have, a win condition chess does not use. If the engine ever
grows a chess assumption, this file is where it shows up.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator, PieceValues, mean_abs_non_royal
from chess_variant_sandbox.agents.features import VOCABULARY
from chess_variant_sandbox.agents.heuristics import Heuristic, register_heuristic
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.tournament.play import play_game


def _only(**terms: float) -> tuple[float, ...]:
    """A weight vector named by its terms, at whatever length the vocabulary now is.

    Written positionally, these two arms broke when `facing_alignment` was added -- and a
    positional list one entry short is worse than a break, because every term after the
    gap has quietly shifted and the arm goes on running.
    """
    weights = [0.0] * len(VOCABULARY)
    for term, weight in terms.items():
        weights[VOCABULARY.index(term)] = weight
    return tuple(weights)

CORE = Path(__file__).resolve().parents[1] / "rust" / "src" / "game.rs"


def build(**overrides) -> dict:
    spec = {
        "schema_version": 1,
        "name": "test",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]}
        ],
        "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}]},
        "rules": {"win": "checkmate"},
    }
    spec.update(overrides)
    return spec


class TestNonSquareBoards:
    @pytest.mark.parametrize(("width", "height"), [(3, 3), (5, 8), (12, 4), (16, 16)])
    def test_boards_of_any_supported_shape_compile_and_move(self, width: int, height: int) -> None:
        spec = build(board={"width": width, "height": height})
        engine = Engine(spec)
        state = engine.initial_state()
        assert engine.spec.n_squares == width * height
        assert engine.legal_moves(state)

    def test_pieces_do_not_wrap_around_the_file_edge(self) -> None:
        """The classic off-by-one in square-index movement: a1 reaching h1 by wrapping."""
        spec = build(
            board={"width": 5, "height": 5},
            setup={"mirror": False, "placements": [
                {"piece": "K", "player": 0, "squares": ["a1"]},
                {"piece": "K", "player": 1, "squares": ["e5"]},
            ]},
        )
        engine = Engine(spec)
        targets = {text[-2:] for text in engine.move_texts(engine.initial_state())}
        assert targets == {"a2", "b1", "b2"}  # a corner king has exactly three squares


class TestHopMovement:
    """Cannon-style movement: jump exactly one screening piece, then land or capture."""

    def cannon_spec(self) -> dict:
        return build(
            board={"width": 5, "height": 5},
            pieces=[
                {"id": "K", "royal": True, "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
                {"id": "C", "value": 5, "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
                    {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
                ]},
            ],
            setup={"mirror": True, "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "C", "squares": ["a1"]},
            ]},
            rules={"win": "royal_capture"},
        )

    def test_a_hop_needs_a_screen(self) -> None:
        engine = Engine(self.cannon_spec())
        state = engine.make_state({"a1": ("C", 0), "a4": ("K", 1), "e1": ("K", 0)})
        assert "Ca1xa4" not in engine.move_texts(state)

    def test_a_hop_captures_over_exactly_one_screen(self) -> None:
        engine = Engine(self.cannon_spec())
        state = engine.make_state(
            {"a1": ("C", 0), "a2": ("K", 0), "a4": ("K", 1), "e1": ("K", 0)}
        )
        assert "Ca1xa4" in engine.move_texts(state)

    def test_two_screens_block_the_hop(self) -> None:
        engine = Engine(self.cannon_spec())
        state = engine.make_state(
            {"a1": ("C", 0), "a2": ("K", 0), "a3": ("K", 0), "a4": ("K", 1), "e1": ("K", 0)}
        )
        assert "Ca1xa4" not in engine.move_texts(state)

    def checkmate_cannon_spec(self) -> dict:
        """Cannons plus a piece that can step into a firing line, under checkmate rules."""
        return build(
            board={"width": 5, "height": 5},
            pieces=[
                {"id": "K", "royal": True, "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
                ]},
                {"id": "A", "value": 2, "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 1}
                ]},
                {"id": "C", "value": 5, "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
                    {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
                ]},
            ],
            setup={"mirror": True, "placements": [
                {"piece": "K", "squares": ["c1"]},
                {"piece": "A", "squares": ["b1"]},
                {"piece": "C", "squares": ["a1"]},
            ]},
            rules={"win": "checkmate"},
        )

    def test_moving_into_a_hop_line_is_an_illegal_self_check(self) -> None:
        """The case the legality shortcut must not miss.

        For sliders, adding a piece to a line can only ever *block* an attack, so the
        filter can skip any move that does not vacate a square. Hops invert that:
        stepping onto an empty square hands an enemy cannon the screen it needs, and
        the move self-checks. An optimisation that only looks at a move's origin
        silently makes such moves legal.
        """
        engine = Engine(self.checkmate_cannon_spec())
        state = engine.make_state(
            {"a4": ("K", 0), "b2": ("A", 0), "a1": ("C", 1), "e5": ("K", 1)}
        )
        assert not engine.in_check(state, player=0)
        moves = engine.move_texts(state)
        assert "Ab2-a2" not in moves, "stepping onto a2 gives the cannon its screen"
        assert "Ab2-b3" in moves, "moves away from the firing line stay legal"

    def test_vacating_a_hop_screen_can_be_legal(self) -> None:
        """The mirror image: removing a screen switches a hop attack *off*."""
        engine = Engine(self.checkmate_cannon_spec())
        state = engine.make_state(
            {"a4": ("K", 0), "a2": ("A", 0), "a1": ("C", 1), "e5": ("K", 1)}
        )
        assert engine.in_check(state, player=0)  # the cannon fires over a2
        assert "Aa2-b2" in engine.move_texts(state)  # stepping aside ends the check

    def test_a_hop_threat_counts_as_an_attack(self) -> None:
        """Attack detection and move generation must agree, or check is unenforceable."""
        engine = Engine(self.cannon_spec())
        state = engine.make_state({"a1": ("C", 1), "a2": ("K", 1), "a4": ("K", 0)})
        assert engine.in_check(state, player=0)


class TestWinConditions:
    def test_royal_capture_ends_when_royalty_is_taken(self) -> None:
        spec = build(rules={"win": "royal_capture"})
        engine = Engine(spec)
        state = engine.make_state({"c1": ("K", 0)}, side_to_move=1)
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_captured"

    def test_royal_capture_does_not_filter_self_check(self) -> None:
        """Under royal_capture, walking into danger is legal -- and losing is the penalty.

        This is the cheap win condition for exploratory sweeps, and the speed comes
        precisely from skipping the per-move legality filter.
        """
        spec = build(rules={"win": "royal_capture"})
        engine = Engine(spec)
        state = engine.make_state({"a1": ("K", 0), "c2": ("K", 1)})
        assert "Ka1-b1" in engine.move_texts(state)  # adjacent to the enemy king

    def test_annihilation_ends_when_a_side_is_empty(self) -> None:
        spec = build(
            pieces=[{"id": "S", "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]}],
            setup={"mirror": True, "placements": [{"piece": "S", "squares": ["c1"]}]},
            rules={"win": "annihilation"},
        )
        engine = Engine(spec)
        state = engine.make_state({"c1": ("S", 0)}, side_to_move=1)
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "annihilated"

    @pytest.mark.parametrize(
        ("setting", "winner"), [("draw", None), ("loss", 0), ("win", 1)]
    )
    def test_stalemate_is_configurable(self, setting: str, winner: int | None) -> None:
        spec = build(rules={"win": "checkmate", "stalemate": setting})
        engine = Engine(spec)
        # Black king on a corner, boxed in by a white king two files away is not enough;
        # build the stalemate explicitly with a rook-like piece.
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
                {"id": "R", "value": 5, "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
            ],
            setup={"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}, {"piece": "R", "squares": ["a1"]}]},
            rules={"win": "checkmate", "stalemate": setting},
        )
        engine = Engine(spec)
        state = engine.make_state(
            {"a5": ("K", 1), "b3": ("R", 0), "c4": ("R", 0), "e1": ("K", 0)}, side_to_move=1
        )
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == winner


class TestMoveModes:
    def test_move_only_cannot_capture(self) -> None:
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
                {"id": "M", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"}]},
            ],
            setup={"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}, {"piece": "M", "squares": ["a1"]}]},
            rules={"win": "royal_capture"},
        )
        engine = Engine(spec)
        state = engine.make_state({"a1": ("M", 0), "a2": ("M", 1), "c1": ("K", 0), "c5": ("K", 1)})
        assert not [t for t in engine.move_texts(state) if t.startswith("Ma1")]

    def test_capture_only_cannot_move_to_an_empty_square(self) -> None:
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
                {"id": "X", "moves": [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "capture"}]},
            ],
            setup={"mirror": True, "placements": [{"piece": "K", "squares": ["c1"]}, {"piece": "X", "squares": ["a1"]}]},
            rules={"win": "royal_capture"},
        )
        engine = Engine(spec)
        state = engine.make_state({"a1": ("X", 0), "c1": ("K", 0), "c5": ("K", 1)})
        assert not [t for t in engine.move_texts(state) if t.startswith("Xa1")]


class TestGeneratorHygiene:
    def test_overlapping_descriptors_do_not_double_generate(self) -> None:
        """Two descriptors covering the same direction must merge, not duplicate.

        Duplicates would not crash anything -- they would quietly inflate branching
        factor, which is an input to the interestingness index.
        """
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 1},
                    {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": 1, "mode": "capture"},
                    {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"},
                ]}
            ],
        )
        engine = Engine(spec)
        moves = engine.move_texts(engine.initial_state())
        assert len(moves) == len(set(moves))

    def test_min_range_suppresses_the_shorter_move(self) -> None:
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2},
                    {"kind": "leap", "vectors": [[1, 0]], "symmetry": "lateral"},
                ]}
            ],
            rules={"win": "royal_capture"},
        )
        engine = Engine(spec)
        state = engine.make_state({"c1": ("K", 0), "a5": ("K", 1)})
        forward = sorted(t for t in engine.move_texts(state) if t.startswith("Kc1-c"))
        assert forward == ["Kc1-c3"]  # two squares only, and the path was clear

    def test_min_range_still_requires_a_clear_path(self) -> None:
        spec = build(
            pieces=[
                {"id": "K", "royal": True, "moves": [
                    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2},
                    {"kind": "leap", "vectors": [[1, 0]], "symmetry": "lateral"},
                ]}
            ],
            rules={"win": "royal_capture"},
        )
        engine = Engine(spec)
        blocked = engine.make_state({"c1": ("K", 0), "c2": ("K", 0), "a5": ("K", 1)})
        assert not [t for t in engine.move_texts(blocked) if t.startswith("Kc1-c")]


class TestCaptureIsARelationAndNotAHierarchy:
    """Nothing below the DSL may special-case what a capture list contains.

    The failure this guards is subtle and specific. A capture matrix is easy to
    implement as though it were a value order -- "a piece may take anything cheaper" --
    because every example anyone reaches for first is one: hierarchy chess is, and so is
    the intuition that a queen beats a pawn. It is not one. The design the feature exists
    for is a *cycle*, where no piece is strongest and the relation is not transitive, and
    an engine that assumed otherwise would play every cyclic game slightly wrong while
    passing every perft, because a perft counts moves and both readings generate moves.
    """

    def cycle(self, **overrides) -> dict:
        spec = build(
            board={"width": 5, "height": 5},
            pieces=[
                {"id": "R", "captures": ["S"], "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
                {"id": "P", "captures": ["R"], "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
                {"id": "S", "captures": ["P"], "moves": [
                    {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            ],
            setup={"mirror": True, "placements": [
                {"piece": "R", "squares": ["a1"]},
                {"piece": "P", "squares": ["c1"]},
                {"piece": "S", "squares": ["e1"]},
            ]},
            rules={"win": "annihilation"},
        )
        spec["schema_version"] = 3
        spec.update(overrides)
        return spec

    def test_no_piece_is_strongest_and_the_engine_does_not_mind(self) -> None:
        """Three pieces, identical movement, and a relation with no top element."""
        from chess_variant_sandbox.engine.moves import move_from, move_to

        engine = Engine(self.cycle())
        square = engine._square_index
        # Each of the three, in turn, surrounded by one of each enemy type.
        for attacker, prey, safe_from, its_own in (
            ("R", "S", "P", "R"), ("P", "R", "S", "P"), ("S", "P", "R", "S"),
        ):
            state = engine.make_state(
                {"c3": (attacker, 0), "c4": (prey, 1), "b3": (safe_from, 1),
                 "d3": (its_own, 1)},
                side_to_move=0,
            )
            targets = {
                move_to(m) for m in engine.legal_moves(state)
                if move_from(m) == square("c3")
            }
            assert square("c4") in targets, f"{attacker} must take {prey}"
            assert square("b3") not in targets, f"{attacker} must not take {safe_from}"
            assert square("d3") not in targets, f"{attacker} must not take its own type"

    def test_the_evaluator_prices_a_cycle_evenly(self) -> None:
        """Three pieces that beat exactly one thing each are worth the same.

        The relational factor is a *share of the enemy army*, so a symmetric army makes
        it symmetric. An implementation that read the capture list as a rank -- "takes
        one type, so it is the weakest" -- would order them, and the ordering would be
        an artefact of the id order rather than of the game.
        """
        from chess_variant_sandbox.agents.evaluation import Evaluator

        values = Evaluator.for_engine(Engine(self.cycle())).values
        assert len(set(round(v, 9) for v in values)) == 1, values

    def test_an_empty_capture_list_is_not_the_same_as_no_capture_list(self) -> None:
        """Absent means everything; empty means nothing. The two are different rules and
        a truthiness test would collapse them into one."""
        from chess_variant_sandbox.engine.compile import compile_spec

        blocker = self.cycle()
        blocker["pieces"][0]["captures"] = []
        compiled = compile_spec(blocker)
        assert not any(compiled.capture_allowed[0])

        free = self.cycle()
        del free["pieces"][0]["captures"]
        assert all(compile_spec(free).capture_allowed[0])

    def test_nothing_in_the_engine_or_the_agents_names_a_capture_list(self) -> None:
        """The chess-assumption test, applied to the newest feature.

        `captures` is a piece key, so it belongs to `spec/` and to `engine/compile.py`,
        which is the one place the DSL is interpreted. Anything below that -- the
        façade, the agents, the evaluator -- must see only the compiled matrix, or the
        rule has two definitions and they will disagree.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "chess_variant_sandbox"
        allowed = {root / "engine" / "compile.py"}
        offenders = []
        for path in sorted((root / "engine").rglob("*.py")) + sorted(
            (root / "agents").rglob("*.py")
        ):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                if '"captures"' in line or "'captures'" in line:
                    offenders.append(f"{path.name}: {stripped}")
        assert not offenders, offenders


class TestNothingInTheCoreWorksOutASquare:
    """Dissolution's named enemy, enumerated, so a fourth instance cannot ship quietly.

    `_compile_square` is the only code in the project that decides where a step lands.
    Everything downstream reads the tables it produces and gets every geometric rule for
    free, for ever. **Everything that computes `rank * width + file` for itself is blind
    to whatever the compiler just dissolved** -- and it fails silently, because a rule
    that changes no move count moves no perft.

    It has bitten three times in two bumps, and each time the site was found by a person
    thinking to ask rather than by anything failing:

    | site | broken by | when | how it ended |
    |---|---|---|---|
    | en passant's midpoint, `apply` halving the move's delta | `wrap` | schema 5 | became the `ep_passes` table |
    | `royals_face`, walking one file by `+= width` | `wrap` | schema 5 | refused, then became the `facing_lines` table |
    | `royals_face` again | `obstacles` | schema 6 | refused, then the same table |
    | `generate_castling`, walking a rank by file index | `obstacles` | schema 6 | **shipped wrong**; the king stood on a wall |

    `reference/schema-6.md` named the *pattern* -- "geometry stopped being what the
    arithmetic says it is" -- and did not enumerate the sites, which is exactly how the
    fourth one shipped. So this test enumerates them, and it is deliberately the ugly
    kind: it reads the source. The alternative is another silent bump.

    **What each survivor is blind to**, which is the half a list of names would not say:

    * `generate_castling` / `castling_rook_squares` / `castling_partner` walk a rank by
      file index. Blind to **walls** -- held shut by validation, since schema 6, because
      a wall on a castling rank is now refused by name. Not blind to a seam: castling is
      stated as a span of one rank and deliberately does not wrap, and the prose says so.
    * `generate_drops`'s `no_doubled` buckets squares by `sq % width`. Correct today
      under both seams and walls -- a file is a file however its ends are joined, and a
      wall holds no piece to be doubled with. Listed so that the next bump *checks* it
      rather than assuming it.

    **Schema 8 found a fifth site outside this test's reach, and it is the renderer.** A
    piece with a facing has to be *drawn* pointing or the game cannot be played by a
    person, and the obvious rotation is `facing * 90` -- which is wrong, because a
    mirrored army's vectors are reflected and a reflection reverses handedness. It is
    `CompiledSpec.facing_headings()`, computed by putting the forward vector through the
    compiler's own two transforms. This test reads `rust/src/game.rs` and cannot see a
    drawing surface; the entry is here so that the next bump knows the class has members
    outside the core.
    """

    # The three functions allowed to do arithmetic on the board's shape, and nothing
    # else. Adding a name here is a deliberate act with a paragraph owed above it.
    ALLOWED = frozenset({
        "generate_castling", "castling_rook_squares", "castling_partner", "generate_drops",
    })

    @staticmethod
    def _sites() -> dict[str, list[str]]:
        """Every line of the core that mentions the board's width, by function.

        Comments are stripped first: this file *discusses* the invariant at length, and a
        test that counted its own explanation would be unreadable and would fail on a
        rewording.
        """
        found: dict[str, list[str]] = {}
        current = "<file scope>"
        for line in CORE.read_text(encoding="utf-8").splitlines():
            name = re.search(r"\bfn\s+([a-z_0-9]+)\s*[(<]", line)
            if name:
                current = name.group(1)
            code = line.split("//")[0]
            if re.search(r"\bwidth\b", code):
                found.setdefault(current, []).append(line.strip())
        return found

    def test_only_the_named_functions_compute_a_square(self) -> None:
        sites = self._sites()
        assert sites, "the pattern matched nothing at all -- the test has gone inert"
        unexpected = set(sites) - self.ALLOWED
        assert not unexpected, (
            "a new place in rust/src/game.rs works a square out by arithmetic: "
            + ", ".join(sorted(unexpected))
            + ". Everything the compiler dissolves -- a seam, a wall, and whatever the "
            "next bump adds -- is invisible to it, and no perft can see the difference. "
            "Read it as a table or say why it cannot be one, and add it to ALLOWED with "
            "a note saying what it is blind to."
        )

    def test_the_enumeration_has_not_gone_stale_the_other_way(self) -> None:
        """A name that stopped doing geometry has to leave the list.

        The list is only worth reading if it is exact. `royals_face` was on it until it
        became a table, and leaving it behind would have made the next reader look for a
        defect that is not there.
        """
        assert set(self._sites()) == self.ALLOWED

    def test_the_flying_general_is_a_table_and_not_a_walk(self) -> None:
        """The one this item retired, pinned so it cannot come back as arithmetic.

        Named separately from the enumeration because the enumeration would still pass if
        `royals_face` recomputed the file some other way -- by `n_squares`, say.
        """
        source = CORE.read_text(encoding="utf-8")
        body = source[source.index("fn royals_face"):]
        body = body[: body.index("\n    }")]
        assert "facing_from" in body and "facing_path" in body
        for arithmetic in ("width", "n_squares", "height"):
            assert arithmetic not in body.replace("`+= width`", ""), arithmetic


class TestAPlayersPieceValuesAssumeNoPieces:
    """Stage 30 added a second per-player vector whose length is the piece count.

    That is the first thing in the project shaped like the ruleset rather than like the
    vocabulary, which makes it the most tempting place yet for a chess assumption --
    "six types", "the last one is royal", "the first one is the cheapest". None of those
    is a rule the schema declares.
    """

    def spec(self, types: int, royal: int | None = 0) -> dict:
        """A board of ``types`` interchangeable pieces, optionally one of them royal.

        Interchangeable on purpose: the derived table is then flat, so anything that
        happened to work because chess's values happen to be distinct fails here.
        """
        pieces = []
        for index in range(types):
            piece = {
                "id": chr(ord("a") + index),
                "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}],
            }
            if index == royal:
                piece["royal"] = True
            pieces.append(piece)
        # Laid out across ranks rather than one per file, because the widest board the
        # schema allows is narrower than the most piece types it allows.
        squares = ["abcdefgh"[i % 8] + str(i // 8 + 1) for i in range(types)]
        return {
            "schema_version": 1,
            "name": "test",
            "board": {"width": 8, "height": 8},
            "pieces": pieces,
            "setup": {
                "mirror": True,
                "placements": [
                    {"piece": p["id"], "squares": [s]}
                    for p, s in zip(pieces, squares, strict=True)
                ],
            },
            "rules": {"win": "checkmate" if royal is not None else "annihilation"},
        }

    @pytest.mark.parametrize("types", [1, 2, 7, 13, 24])
    def test_a_valuation_is_as_long_as_the_ruleset_says(self, types: int) -> None:
        """One type up to the validator's own cap, royal and not.

        `MAX_PIECE_TYPES` is 24 and `Heuristic` bounds a stored table by it, so 24 is
        the widest a table can legitimately be. One is the narrowest, and with the only
        piece royal there is no non-royal piece to take a mean over at all -- which is
        the division the scale rule would fail on if it did not floor.
        """
        engine = Engine(self.spec(types, royal=None))
        derived = Evaluator.for_spec(engine.spec).values
        assert len(derived) == types
        pair = PieceValues.of(derived, tuple(range(1, types + 1)), engine.spec.royal_types)
        assert len(pair.objective) == types
        assert mean_abs_non_royal(
            pair.objective, engine.spec.royal_types
        ) == pytest.approx(mean_abs_non_royal(derived, engine.spec.royal_types))

    def test_a_ruleset_of_nothing_but_royals_leaves_nothing_to_price(self) -> None:
        """The degenerate end, which must not divide by nothing.

        Where every piece is royal there is no non-royal mean, so there is no scale to
        put an override on -- and every slot an override could fill is one it may not.
        Pricing a royal is refused; a table of zeros is refused by ``Heuristic``, which
        is where an override that says nothing can still be caught without a board; and
        a table of zeros that reaches the pair anyway is *inert* rather than a crash,
        because a royal is worth zero either way.
        """
        engine = Engine(self.spec(1, royal=0))
        derived = Evaluator.for_spec(engine.spec).values
        assert engine.spec.royal_types == (0,)
        with pytest.raises(ValueError, match="royal"):
            PieceValues.of(derived, (1.0,), engine.spec.royal_types)
        with pytest.raises(Exception, match="absence of one"):
            Heuristic(
                name="x",
                plan="",
                weights=_only(material=1.0),
                values=(0.0,),
            )
        assert PieceValues.of(derived, (0.0,), engine.spec.royal_types).objective == (0.0,)

    @pytest.mark.parametrize("royal", [0, 2, 4])
    def test_which_slot_is_royal_is_read_off_the_rules(self, royal: int) -> None:
        """Not "the last one", which is only true of chess and of most things like it."""
        engine = Engine(self.spec(5, royal=royal))
        derived = Evaluator.for_spec(engine.spec).values
        assert engine.spec.royal_types == (royal,)
        priced = [1.0] * 5
        priced[royal] = 0.0
        assert PieceValues.of(derived, tuple(priced), engine.spec.royal_types).objective[
            royal
        ] == 0.0
        priced[royal] = 1.0
        with pytest.raises(ValueError, match=f"piece type {royal} is royal"):
            PieceValues.of(derived, tuple(priced), engine.spec.royal_types)

    def test_a_priced_player_plays_a_game_of_any_piece_count(self) -> None:
        """The end-to-end form: the override crosses the boundary and a game happens.

        The boundary carries the table raw and the core checks its length against the
        tables it holds, so a ruleset with three types and one with thirteen exercise
        two different lengths of the same code path.
        """
        for types in (3, 13):
            engine = Engine(self.spec(types, royal=None))
            weights = _only(material=1.0, mobility=0.05)
            heuristic = Heuristic(
                name=f"priced-{types}",
                plan="",
                weights=weights,
                values=tuple(float(index + 1) for index in range(types)),
            )
            register_heuristic(heuristic)
            record = play_game(
                engine, (f"greedy:h={heuristic.digest}", "random"), seed=1
            )
            assert record.plies > 0
