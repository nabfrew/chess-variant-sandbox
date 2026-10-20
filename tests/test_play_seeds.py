"""A game nobody named a seed for gets a fresh one, and keeps it (Q987).

Every game on either server used to run with seed 0 when the request named none, so the
computer chose the same move in the same position at the same ply in every game anyone
played (`docs/reference/round-hardest-weakness.md` §1, §7). `play.seed_asked` is the one
place a request's seed is read; `Sessions.start` draws a fresh one when given none. The
store's side, the match row, is `tests/test_site_matches.py::TestTheSeed`.

Nothing here works out a move: the core's `random` player chooses, and the tests read
which move it chose under which seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library
from chess_variant_sandbox.web.play import SEEDS, Sessions, fresh_seed, seed_asked

DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))
GAME = library.Game(
    hash=spec_hash(DULL),
    name=str(DULL["name"]),
    description="",
    path=Path(),
    source="corpus",
    spec=DULL,
)


def _first_reply(sessions: Sessions, seed: int | None) -> tuple[int, int]:
    """The seed the session holds and the move the core's `random` player opened with."""
    session = sessions.start(GAME, ("random", "random"), seed=seed)
    snapshot = session.play_agent()
    return session.seed, int(session.played[0]) if snapshot["history"] else -1


class TestARequestsSeed:
    def test_none_named_is_a_fresh_one_every_time(self) -> None:
        drawn = [seed_asked(None) for _ in range(8)]
        assert len(set(drawn)) == len(drawn), "two requests drew the same seed"
        assert all(0 <= seed < SEEDS for seed in drawn)

    @pytest.mark.parametrize("named", [0, 1, 987654321, "42"])
    def test_a_named_seed_is_kept_exactly_and_zero_is_one_like_any_other(
        self, named: object
    ) -> None:
        """0 is a seed somebody chose, not a sentinel for *none*."""
        assert seed_asked(named) == int(str(named))

    @pytest.mark.parametrize("junk", [True, "fresh", 1.5, [3]])
    def test_anything_that_is_not_a_whole_number_is_refused_not_guessed(
        self, junk: object
    ) -> None:
        with pytest.raises(ValueError, match="whole number"):
            seed_asked(junk)

    def test_the_draw_is_below_the_bound_the_store_and_core_take(self) -> None:
        assert all(0 <= fresh_seed() < SEEDS for _ in range(32))
        assert SEEDS == 2**31


class TestASessionsSeed:
    def test_two_new_sessions_get_different_seeds(self) -> None:
        sessions = Sessions()
        first = sessions.start(GAME, ("human", "random"))
        second = sessions.start(GAME, ("human", "random"))
        assert first.seed != second.seed

    def test_a_named_seed_is_the_sessions_seed(self) -> None:
        sessions = Sessions()
        assert sessions.start(GAME, ("human", "random"), seed=0).seed == 0
        assert sessions.start(GAME, ("human", "random"), seed=31337).seed == 31337

    def test_the_computers_move_is_a_function_of_the_seed_it_was_given(self) -> None:
        """The same seed opens with the same move, and the seed is what varies it.

        So a fresh seed is what stops every game from opening the same way, and the seed
        kept with the session is what replays the game exactly.
        """
        sessions = Sessions()
        seed, reply = _first_reply(sessions, None)
        assert _first_reply(sessions, seed) == (seed, reply)
        openings = {_first_reply(sessions, named)[1] for named in range(12)}
        assert len(openings) > 1, "the computer opened the same way under twelve seeds"
