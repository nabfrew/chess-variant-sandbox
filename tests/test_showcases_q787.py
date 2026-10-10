"""Q787: the demonstration tests of the three showcases the 2026-10-09 audit found missing.

Gaps G1 to G3 of `docs/reference/schema-showcase-coverage.md` (*The 2026-10-09 audit*):
`promote_on_entry`, the entering-play half of `facing.initial`, and the `contacts` spelling
of `position: "choose"`. Each class below tests the prediction the document pre-registered
under *Q787* before this file was written, item by item; the numbering is the document's.

A test here drives the core through `Engine` and asserts what the core did. The only choice
of a move in Python is the seeded random walk of clause 4, which is the audit's probe and
not an agent.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from pathlib import Path

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_NORMAL, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]
GAMES = 6  # seeds 0 to 5, as pre-registered
PLIES = 250


def _engine(path: str) -> Engine:
    return Engine(load_spec(ROOT / path))


def _walk(engine: Engine, seed: int) -> Iterator[tuple[GameState, int]]:
    """A seeded uniformly random game: each (position, move played) in order."""
    rng = random.Random(seed)
    state = engine.initial_state()
    for _ in range(PLIES):
        if engine.terminal(state) is not None:
            return
        moves = engine.legal_moves(state)
        if not moves:
            return
        move = rng.choice(moves)
        yield state, move
        state = engine.apply(state, move)


def _with_hand(engine: Engine, state: GameState, player: int, **held: int) -> GameState:
    """``state`` with ``held`` (piece id -> count) in ``player``'s hand, key recomputed."""
    spec = engine.spec
    hands = [tuple([0] * spec.n_types), tuple([0] * spec.n_types)]
    hand = [0] * spec.n_types
    for piece, count in held.items():
        hand[spec.piece_ids.index(piece)] = count
    hands[player] = tuple(hand)
    moved = state.with_(hands=(hands[0], hands[1]), side_to_move=player)
    key = engine._compute_hash(moved)
    return moved.with_(key=key, repetition=(key,))


def _occupant(engine: Engine, state: GameState, square: str) -> str | None:
    value = state.board[engine._square_index(square)]
    return None if value == 0 else engine.spec.piece_ids[engine.spec.type_of(value)]


def _count(engine: Engine, state: GameState, player: int) -> int:
    return sum(1 for v in state.board if v and engine.spec.owner_of(v) == player)


def _perfts(engine: Engine, depth: int) -> list[int]:
    start = engine.initial_state()
    return [engine.perft(start, d) for d in range(1, depth + 1)]


# ---------------------------------------------------------------------------- G1


class TestShogiDropPromoted:
    """`archive/shogi-drop-promoted`: shogi plus `rules.promote_on_entry` and nothing else."""

    @pytest.fixture(scope="module")
    def game(self) -> Engine:
        return _engine("archive/shogi-drop-promoted.json")

    @pytest.fixture(scope="module")
    def shogi(self) -> Engine:
        return _engine("rulesets/shogi.json")

    def test_it_is_shogi_plus_one_key(self) -> None:
        mine = load_spec(ROOT / "archive/shogi-drop-promoted.json")
        theirs = load_spec(ROOT / "rulesets/shogi.json")
        assert mine["rules"].pop("promote_on_entry") is True
        for spec in (mine, theirs):
            for key in ("name", "description", "provenance"):
                spec.pop(key)
        assert mine == theirs

    def test_1_a_silver_dropped_in_the_zone_stands_promoted(self, game: Engine, shogi: Engine) -> None:
        # The spec's `ranks: [6, 7, 8]` are 0-based: the zone is ranks 7 to 9 (the far three).
        def after(engine: Engine, text: str) -> str | None:
            kings = engine.make_state({"e1": ("K", 0), "e9": ("K", 1)})
            state = _with_hand(engine, kings, 0, S=1)
            return _occupant(engine, engine.apply(state, engine.find_move(state, text)), text[2:])

        assert after(game, "S@e7") == "Z"
        assert after(game, "S@e6") == "S"
        assert after(shogi, "S@e7") == "S"  # the control: the same drop, no flag

    def test_2_a_pawn_dropped_in_the_zone_stands_as_a_tokin(self, game: Engine) -> None:
        kings = {"e1": ("K", 0), "e9": ("K", 1)}
        white = _with_hand(game, game.make_state(kings), 0, P=1)
        for square, expected in (("e7", "Q"), ("e6", "P"), ("a9", "Q")):
            moved = game.apply(white, game.find_move(white, f"P@{square}"))
            assert _occupant(game, moved, square) == expected, square
        # Black's zone is the mirror: ranks 3, 2 and 1.
        black = _with_hand(game, game.make_state(kings), 1, P=1)
        for square, expected in (("e3", "Q"), ("e4", "P"), ("a1", "Q")):
            moved = game.apply(black, game.find_move(black, f"P@{square}"))
            assert _occupant(game, moved, square) == expected, square

    def test_2_a_pawn_may_be_dropped_on_the_last_rank_only_with_the_flag(
        self, game: Engine, shogi: Engine
    ) -> None:
        """The last rank is a dead end for a pawn in shogi and the flag makes it a tokin."""
        kings = {"e1": ("K", 0), "e9": ("K", 1)}
        in_shogi = _with_hand(shogi, shogi.make_state(kings), 0, P=1)
        flagged = _with_hand(game, game.make_state(kings), 0, P=1)
        assert "P@a9" not in shogi.move_texts(in_shogi)
        assert "P@a9" in game.move_texts(flagged)

    def test_3_the_opening_perfts_are_shogi_s(self, game: Engine, shogi: Engine) -> None:
        assert _perfts(game, 3) == _perfts(shogi, 3)

    def test_4_random_play_drops_promoted_exactly_where_the_rule_says(self, game: Engine) -> None:
        spec = game.spec
        spec_json = load_spec(ROOT / "archive/shogi-drop-promoted.json")
        into = {r["piece"]: spec.piece_ids.index(r["to"][0]) for r in spec_json["rules"]["promotion"]}
        zone = set(spec_json["rules"]["promotion"][0]["ranks"])
        games_with_a_drop = 0
        for seed in range(GAMES):
            dropped = 0
            for state, move in _walk(game, seed):
                origin, target, kind, held = decode_move(move)
                if kind != MOVE_DROP:
                    continue
                dropped += 1
                piece = held % spec.n_types
                player = state.side_to_move
                rank = target // spec.width  # 0-based, as `ranks` is
                own_rank = rank if player == 0 else spec.height - 1 - rank
                value = game.apply(state, move).board[target]
                promotes = own_rank in zone and spec.piece_ids[piece] in into
                expected = into[spec.piece_ids[piece]] if promotes else piece
                assert spec.type_of(value) == expected, (seed, game.describe_move(state, move))
            games_with_a_drop += dropped > 0
        assert games_with_a_drop >= 5, games_with_a_drop


# ---------------------------------------------------------------------------- G2


class TestShogiBackwardDrops:
    """`archive/shogi-backward-drops`: a dropped pawn arrives facing its own side."""

    @pytest.fixture(scope="module")
    def game(self) -> Engine:
        return _engine("archive/shogi-backward-drops.json")

    @pytest.fixture(scope="module")
    def control(self) -> Engine:
        """The clean control: this spec with `initial` deleted (the pawn still faces and turns)."""
        spec = load_spec(ROOT / "archive/shogi-backward-drops.json")
        pawn = next(p for p in spec["pieces"] if p["id"] == "P")
        del pawn["facing"]["initial"]
        assert validate_spec(copy.deepcopy(spec)) == []
        return Engine(spec)

    def test_1_the_opening_is_shogi_s_plus_nine_turns_with_or_without_initial(
        self, game: Engine, control: Engine
    ) -> None:
        assert len(game.legal_moves(game.initial_state())) == 39
        assert len(control.legal_moves(control.initial_state())) == 39
        assert _perfts(game, 3) == _perfts(control, 3)

    def _after_a_drop_and_a_reply(self, engine: Engine) -> tuple[GameState, GameState]:
        kings = {"e1": ("K", 0), "e9": ("K", 1)}
        state = _with_hand(engine, engine.make_state(kings), 0, P=1)
        dropped = engine.apply(state, engine.find_move(state, "P@e5"))
        # Any reply: black has a king and nothing else.
        return dropped, engine.apply(dropped, engine.legal_moves(dropped)[0])

    def test_2_a_dropped_pawn_faces_its_own_side(self, game: Engine, control: Engine) -> None:
        dropped, _ = self._after_a_drop_and_a_reply(game)
        e5 = game._square_index("e5")
        assert game.spec.facing_of(dropped.board[e5]) == 1  # backward
        dropped, _ = self._after_a_drop_and_a_reply(control)
        assert control.spec.facing_of(dropped.board[e5]) == 0  # forward

    def test_3_the_dropped_pawn_steps_back_or_turns(self, game: Engine, control: Engine) -> None:
        for engine, step in ((game, "e4"), (control, "e6")):
            _, replied = self._after_a_drop_and_a_reply(engine)
            e5 = engine._square_index("e5")
            names = sorted(
                engine.describe_move(replied, m)
                for m in engine.legal_moves(replied)
                if decode_move(m)[0] == e5
            )
            assert len(names) == 2, names
            assert any(n.endswith(step) and decode_move(engine.find_move(replied, n))[2] == MOVE_NORMAL
                       for n in names), (step, names)
        _, replied = self._after_a_drop_and_a_reply(game)
        e6 = game._square_index("e6")
        assert not any(
            decode_move(m)[0] == game._square_index("e5") and decode_move(m)[1] == e6
            for m in game.legal_moves(replied)
        )

    def test_4_random_play_drops_every_pawn_backward(self, game: Engine) -> None:
        spec = game.spec
        pawn = spec.piece_ids.index("P")
        games_with_a_drop = 0
        for seed in range(GAMES):
            dropped = 0
            for state, move in _walk(game, seed):
                _, target, kind, held = decode_move(move)
                if kind != MOVE_DROP or held % spec.n_types != pawn:
                    continue
                dropped += 1
                value = game.apply(state, move).board[target]
                assert spec.facing_of(value) == 1, (seed, game.describe_move(state, move))
            games_with_a_drop += dropped > 0
        assert games_with_a_drop >= 5, games_with_a_drop


# ---------------------------------------------------------------------------- G3


class TestPawnKnockdown:
    """`archive/pawn-knockdown`: a pawn's capture knocks a piece down a rank and throws it."""

    @pytest.fixture(scope="module")
    def game(self) -> Engine:
        return _engine("archive/pawn-knockdown.json")

    def _words(self, engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
        wanted = (engine._square_index(origin), engine._square_index(target))
        return [m for m in engine.legal_moves(state) if decode_move(m)[:2] == wanted]

    def test_1_a_pawn_taking_a_bishop_offers_every_landing(self, game: Engine) -> None:
        state = game.make_state({"e4": ("P", 0), "d5": ("B", 1), "e1": ("K", 0), "e8": ("K", 1)})
        words = self._words(game, state, "e4", "d5")
        assert len(words) == 8
        landings = set()
        for word in words:
            after = game.apply(state, word)
            assert _occupant(game, after, "d5") == "P"
            assert after.board[game._square_index("d5")] == game.spec.encode_piece(0, game.spec.piece_ids.index("P"))
            black = [
                (sq, game.spec.type_of(v))
                for sq, v in enumerate(after.board)
                if v and game.spec.owner_of(v) == 1 and sq != game._square_index("e8")
            ]
            assert [t for _, t in black] == [game.spec.piece_ids.index("N")], black
            landings.add(black[0][0])
        assert len(landings) == 8
        # A knight taking the same bishop removes it.
        knight = game.make_state({"b4": ("N", 0), "d5": ("B", 1), "e1": ("K", 0), "e8": ("K", 1)})
        taking = self._words(game, knight, "b4", "d5")
        assert len(taking) == 1
        after = game.apply(knight, taking[0])
        assert _count(game, after, 1) == 1

    @pytest.mark.parametrize(("victim", "left"), [("Q", "R"), ("R", "B"), ("B", "N"), ("N", "P")])
    def test_2_the_chain_knocks_down_one_rank(self, game: Engine, victim: str, left: str) -> None:
        state = game.make_state({"e4": ("P", 0), "d5": (victim, 1), "e1": ("K", 0), "h8": ("K", 1)})
        words = self._words(game, state, "e4", "d5")
        assert words
        for word in words:
            after = game.apply(state, word)
            others = sorted(
                _occupant(game, after, f"{f}{r}")
                for f in "abcdefgh" for r in range(1, 9)
                if _occupant(game, after, f"{f}{r}") and f"{f}{r}" not in ("d5", "e1", "h8")
            )
            assert others == [left], (victim, others)
            assert _count(game, after, 1) == 2

    def test_2_a_pawn_taking_a_pawn_is_an_ordinary_capture(self, game: Engine) -> None:
        state = game.make_state({"e4": ("P", 0), "d5": ("P", 1), "e1": ("K", 0), "h8": ("K", 1)})
        words = self._words(game, state, "e4", "d5")
        assert len(words) == 1
        after = game.apply(state, words[0])
        assert _count(game, after, 1) == 1
        assert _occupant(game, after, "d5") == "P"

    def test_3_the_opening_perfts_are_chess_s(self, game: Engine) -> None:
        # Chess's own, not the published numbers: no pawn can take anything but a pawn before ply 5.
        assert _perfts(game, 3) == _perfts(_engine("rulesets/chess.json"), 3)

    def test_4_random_play_throws_the_piece_and_loses_none(self, game: Engine) -> None:
        spec = game.spec
        pawn = spec.piece_ids.index("P")
        minors = {spec.piece_ids.index(p) for p in "NBRQ"}
        games_with_a_knockdown = 0
        for seed in range(GAMES):
            knocked = 0
            for state, move in _walk(game, seed):
                origin, target, kind, _ = decode_move(move)
                victim = state.board[target]
                mover = state.side_to_move
                if (
                    kind != MOVE_NORMAL
                    or spec.type_of(state.board[origin]) != pawn
                    or not victim
                    or spec.owner_of(victim) == mover
                    or spec.type_of(victim) not in minors
                ):
                    continue
                knocked += 1
                after = game.apply(state, move)
                assert spec.type_of(after.board[target]) == pawn
                assert spec.owner_of(after.board[target]) == mover
                assert _count(game, after, 1 - mover) == _count(game, state, 1 - mover), seed
                assert _count(game, after, mover) == _count(game, state, mover), seed
            games_with_a_knockdown += knocked > 0
        assert games_with_a_knockdown >= 5, games_with_a_knockdown
