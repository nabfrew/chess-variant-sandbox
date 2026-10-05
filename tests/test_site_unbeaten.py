"""Stars, the three sort keys and the *Unbeaten* tag.

Task 1.5 of `docs/plan-website.md`, against §5. Three games carry the whole file -- one
rated with a clear top, one rated with a tied top, and one with no picker ladder at all --
because every rule the tag has is a statement about which of those three a result lands on
and which rows it is allowed to read.

What the acceptance asks for is here by name: each ordering; that a sandbox win, a draw
and a win against a non-top computer player clear nothing; that a clear survives the top
changing hands; that a tied top needs each of the tied beaten; and that the rows the
loader brought in from the local application clear nothing either.

These need a real Postgres and skip with a reason without one, each inside a scratch
schema it drops -- task 1.2's recipe, since `py-pglite` does not work on Windows.

Nothing here works out a legal move, an outcome or a winner. Where a game is played it is
the engine that ends it; where a result is stated it is stated outright, as the thing the
engine would have said, and never derived.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store

from chess_variant_sandbox.site import browse, unbeaten, votes
from chess_variant_sandbox.site import matches as m
from chess_variant_sandbox.site import rating as r
from chess_variant_sandbox.site.db import connect
from chess_variant_sandbox.site.server import SiteApp
from chess_variant_sandbox.spec import spec_hash
from chess_variant_sandbox.web import library

DULL = json.loads((library.RULESET_DIR / "dull.json").read_text(encoding="utf-8"))

TOP = "minimax:nodes=1600"
MIDDLE = "minimax:nodes=400"
BOTTOM = "random"
RIVAL = "greedy"


def _dull(name: str, max_plies: int) -> dict[str, Any]:
    """`dull` under another name and ply limit, which is a different game and a different
    hash -- the name is outside the mechanics hash, so the limit is what separates them."""
    spec = json.loads(json.dumps(DULL))
    spec["name"] = name
    spec["rules"]["max_plies"] = max_plies
    return spec


# Three games. `ALPHA` has a clear top, `BETA` a tied one, and `GAMMA` no ladder at all,
# which is the state thirty of the library's sixty-one games are in (§3).
ALPHA = _dull("alpha", 2)
BETA = _dull("beta", 4)
GAMMA = _dull("gamma", 6)
ALPHA_HASH = spec_hash(ALPHA)
BETA_HASH = spec_hash(BETA)
GAMMA_HASH = spec_hash(GAMMA)

# A picker ladder in the shape `runs/opponents/<hash>.json` has. Its ratings sum to zero,
# as a fitted one's do; the seeding puts them on the site's scale.
CLEAR_LADDER = {
    "ruleset_hash": ALPHA_HASH,
    "standings": [
        {"name": TOP, "rating": 400.0, "games": 24},
        {"name": MIDDLE, "rating": 0.0, "games": 24},
        {"name": BOTTOM, "rating": -400.0, "games": 24},
    ],
}

# Two within `rating.TOP_BAND` of each other, which is a tie the ladder could not break.
TIED_LADDER = {
    "ruleset_hash": BETA_HASH,
    "standings": [
        {"name": TOP, "rating": 300.0, "games": 24},
        {"name": RIVAL, "rating": 298.0, "games": 24},
        {"name": BOTTOM, "rating": -598.0, "games": 24},
    ],
}


# ------------------------------------------------------------------------ the fixture


class Scratch:
    """A store with the schema in it, three games, and a way to seat people in them."""

    def __init__(self, url: str, schema: str) -> None:
        self.url = url
        self.schema = schema
        self.connection = self.open()
        self.minute = 0

    def open(self) -> Any:
        connection = connect(self.url)
        with connection.cursor() as cursor:
            cursor.execute(f"set search_path to {self.schema}")
        return connection

    def user(self, handle: str) -> int:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into users (handle, email) values (%s, %s) returning id",
                (handle, f"{handle}@example.invalid"),
            )
            found = cursor.fetchone()
        self.connection.commit()
        return int(found[0])

    def game(self, digest: str, spec: dict[str, Any], ladder: Any = None) -> str:
        with self.connection.cursor() as cursor:
            cursor.execute(
                "insert into games (hash, spec, name, origin) values (%s, %s::jsonb, %s, %s)",
                (digest, json.dumps(spec), spec["name"], "corpus"),
            )
        if ladder is not None:
            r.seed(self.connection, digest, ladder)
        self.connection.commit()
        return digest

    def play(
        self,
        game: str,
        user: int,
        agent: str,
        result: float | None = None,
        mode: str = m.LEADERBOARD,
        seat: int = 0,
        source: str | None = None,
    ) -> m.Row:
        """Seat a person, then state how the game ended.

        The row is written by `matches.start`, so the *top* bit and the tied group on it
        are the real ones, read live off the table at that moment. The ending is stated
        rather than played out, because these tests are about which rows the tag reads and
        not about what the engine decides; `tests/test_site_matches.py` is where a game is
        actually finished, and one test below plays one out to the engine's own verdict.
        """
        seats = ["human", agent] if seat == 0 else [agent, "human"]
        row = m.start(self.connection, game, mode, seats, user)
        if source is not None:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    "update matches set source = %s where id = %s", (source, row.id)
                )
        if result is not None:
            self.conclude(row.id, result if seat == 0 else 1.0 - result)
        self.connection.commit()
        return m.row(self.connection, row.id)

    def conclude(self, match_id: str, result: float) -> None:
        """Finish a match at a stated minute, and move the table as a finish does.

        `result` is seat 0's score, which is what the column holds. The minute climbs, so
        the order the fold reads and the order a first clear is decided in are both
        stated rather than left to a clock.
        """
        self.minute += 1
        with self.connection.cursor() as cursor:
            cursor.execute(
                "update matches set result = %s, reason = 'checkmate',"
                " finished = %s where id = %s",
                (result, f"2026-09-11 12:{self.minute:02d}:00+00", match_id),
            )
        r.finish(self.connection, match_id)
        self.connection.commit()

    def lift(self, game: str, agent: str, to: float) -> None:
        """Move a computer player's rating outright, which is how the top changes hands.

        Human play is what moves them for real (§5); this says *the table now reads this*
        without playing the twenty games it would take, because what is under test is the
        tag's behaviour when the top moves and not the arithmetic that moved it.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                "update ratings set rating = %s where game = %s and player_agent = %s",
                (to, game, agent),
            )
        self.connection.commit()


@contextlib.contextmanager
def _scratch() -> Iterator[Scratch]:
    url, name = site_store.emptied("no database to seed three games in")
    scratch = Scratch(url, name)
    # Committed, so the `search_path` survives a test's rollback. `migrate.run` used to
    # commit it in passing, when every test migrated a schema of its own.
    scratch.connection.commit()
    try:
        yield scratch
    finally:
        scratch.connection.close()


@pytest.fixture
def store() -> Iterator[Scratch]:
    """Three games: a clear top, a tied top, and one nobody has rated."""
    with _scratch() as scratch:
        scratch.game(ALPHA_HASH, ALPHA, CLEAR_LADDER)
        scratch.game(BETA_HASH, BETA, TIED_LADDER)
        scratch.game(GAMMA_HASH, GAMMA)
        yield scratch


# --------------------------------------------------------------------------- the votes


class TestVotes:
    def test_a_vote_is_written_and_read_back(self, store: Scratch) -> None:
        ada = store.user("ada")
        votes.cast(store.connection, ada, ALPHA_HASH, 4, ["new", "clever"])
        found = votes.of(store.connection, ada, ALPHA_HASH)
        assert found is not None
        # The tags come back in `votes.TAGS`' order, whatever order they were chosen in.
        assert (found.stars, found.tags) == (4, ("clever", "new"))

    def test_voting_again_replaces_rather_than_adds(self, store: Scratch) -> None:
        """§5: one row each, replaced on re-vote. A person cannot stuff their own box."""
        ada = store.user("ada")
        votes.cast(store.connection, ada, ALPHA_HASH, 1)
        votes.cast(store.connection, ada, ALPHA_HASH, 5)
        assert votes.stars(store.connection, ALPHA_HASH) == votes.Stars(5.0, 1)

    def test_two_people_are_two_rows(self, store: Scratch) -> None:
        ada, bob = store.user("ada"), store.user("bob")
        votes.cast(store.connection, ada, ALPHA_HASH, 2)
        votes.cast(store.connection, bob, ALPHA_HASH, 4)
        assert votes.stars(store.connection, ALPHA_HASH) == votes.Stars(3.0, 2)

    def test_a_vote_off_the_scale_is_refused_by_name(self, store: Scratch) -> None:
        ada = store.user("ada")
        for stars in (0, 6, -1):
            with pytest.raises(votes.VoteError) as error:
                votes.cast(store.connection, ada, ALPHA_HASH, stars)
            assert "1 to 5 stars" in str(error.value)

    def test_an_unvoted_game_has_no_mean_rather_than_a_zero(self, store: Scratch) -> None:
        """An unvoted game is not a game everybody hated, and a sort that read it as one
        would put every new game last."""
        found = votes.stars(store.connection, ALPHA_HASH)
        assert found.mean is None and found.count == 0 and not found.voted

    def test_a_vote_can_be_taken_back(self, store: Scratch) -> None:
        ada = store.user("ada")
        votes.cast(store.connection, ada, ALPHA_HASH, 3)
        assert votes.withdraw(store.connection, ada, ALPHA_HASH)
        assert votes.of(store.connection, ada, ALPHA_HASH) is None
        assert not votes.withdraw(store.connection, ada, ALPHA_HASH)


# ----------------------------------------------------------------------------- the tag


class TestTheTag:
    def test_a_rated_game_starts_unbeaten_and_names_its_target(
        self, store: Scratch
    ) -> None:
        tag = unbeaten.tag(store.connection, ALPHA_HASH)
        assert tag.rated and tag.unbeaten is True
        assert tag.top == (TOP,)
        assert tag.first is None and tag.clears == 0

    def test_an_unrated_game_carries_no_tag_at_all(self, store: Scratch) -> None:
        """§3 found thirty of the sixty-one library games in this state. *Unrated* is not
        *unbeaten*: there is nothing to beat, so the answer is neither true nor false."""
        tag = unbeaten.tag(store.connection, GAMMA_HASH)
        assert not tag.rated
        assert tag.unbeaten is None
        assert tag.top == () and tag.top_rating is None

    def test_beating_the_top_clears_it_and_names_the_first(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)

        tag = unbeaten.tag(store.connection, ALPHA_HASH)
        assert tag.unbeaten is False
        assert tag.first is not None
        assert (tag.first.handle, tag.first.seat, tag.first.agent) == ("ada", 0, TOP)
        assert tag.clears == 1

    def test_a_win_from_the_second_seat_clears_it_too(self, store: Scratch) -> None:
        """The user, 2026-09-10: one win from either seat clears, and the seat is
        recorded. The lever §5 names and does not pull is requiring both."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0, seat=1)

        tag = unbeaten.tag(store.connection, ALPHA_HASH)
        assert tag.unbeaten is False
        assert tag.first is not None and tag.first.seat == 1

    def test_the_rest_are_counted_after_the_first(self, store: Scratch) -> None:
        ada, bob = store.user("ada"), store.user("bob")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.play(ALPHA_HASH, bob, TOP, result=1.0)

        tag = unbeaten.tag(store.connection, ALPHA_HASH)
        assert tag.first is not None and tag.first.handle == "ada"
        assert tag.clears == 2

    def test_one_person_winning_twice_is_one_clear(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        assert unbeaten.tag(store.connection, ALPHA_HASH).clears == 1

    def test_a_sandbox_win_clears_nothing(self, store: Scratch) -> None:
        """§4: sandbox is read by no statistic, no rating, no anchor row and no sort key.
        The row is not even written with a *top* bit."""
        ada = store.user("ada")
        row = store.play(ALPHA_HASH, ada, TOP, result=1.0, mode=m.SANDBOX)

        assert not row.top and row.top_players == ()
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True

    def test_a_draw_clears_nothing(self, store: Scratch) -> None:
        """The user's word is *beaten*, and a draw is half a point each."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=0.5)
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True

    def test_a_loss_clears_nothing(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=0.0)
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True

    def test_a_win_against_a_non_top_computer_player_clears_nothing(
        self, store: Scratch
    ) -> None:
        """*Won* and *beat the top* are two different states, and `random` is the reason
        §5 keeps them apart: the rung says which one, and a win over `random` reads as
        what it is."""
        ada = store.user("ada")
        middle = store.play(ALPHA_HASH, ada, MIDDLE, result=1.0)
        bottom = store.play(ALPHA_HASH, ada, BOTTOM, result=1.0)

        assert not middle.top and not bottom.top
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True

    def test_a_clear_survives_the_top_changing_hands(self, store: Scratch) -> None:
        """§5: a clear is a record, decided at the time, and it stays.

        The bit and the group are written at the start, so a computer player rated later
        and landing on top cannot make a beaten game unbeaten again. The live target moves
        and the tag does not.
        """
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is False

        store.lift(ALPHA_HASH, MIDDLE, to=3000.0)

        tag = unbeaten.tag(store.connection, ALPHA_HASH)
        assert tag.top == (MIDDLE,), "the live target did not move"
        assert tag.unbeaten is False, "a cleared game came back as unbeaten"
        assert tag.first is not None and tag.first.agent == TOP

    def test_a_tied_top_needs_each_of_the_tied_beaten(self, store: Scratch) -> None:
        """§5: a tie the ladder could not break stays a tie until a match breaks it, and
        clearing a tied game means beating each of the tied.

        This is the rule one boolean cannot carry, and why `matches.top_players` exists:
        a win over one of two tied computer players has the same *top* bit as a win over
        a lone top, and only the group on the row tells them apart.
        """
        ada = store.user("ada")
        assert set(unbeaten.tag(store.connection, BETA_HASH).top) == {TOP, RIVAL}

        first = store.play(BETA_HASH, ada, TOP, result=1.0)
        assert first.top and set(first.top_players) == {TOP, RIVAL}
        assert unbeaten.tag(store.connection, BETA_HASH).unbeaten is True, (
            "half a tied top cleared the game"
        )

        store.play(BETA_HASH, ada, RIVAL, result=1.0)
        assert unbeaten.tag(store.connection, BETA_HASH).unbeaten is False

    def test_a_tie_is_cleared_by_one_person_and_not_by_two_halves(
        self, store: Scratch
    ) -> None:
        """The clear is a person's. With the tie still standing, two people beating one
        each is two *won* records and no first clear, because neither of them has beaten
        the top.

        The tie is put back between the two matches, because ada's win moves the computer
        player she beat and would otherwise leave the other alone at the top -- at which
        point bob really has beaten the top, and that is a different rule being tested.
        """
        ada, bob = store.user("ada"), store.user("bob")
        seeded = r.load(store.connection, BETA_HASH)[r.computer(TOP)].rating
        store.play(BETA_HASH, ada, TOP, result=1.0)
        store.lift(BETA_HASH, TOP, to=seeded)

        second = store.play(BETA_HASH, bob, RIVAL, result=1.0)

        assert set(second.top_players) == {TOP, RIVAL}, "the tie did not survive"
        tag = unbeaten.tag(store.connection, BETA_HASH)
        assert tag.unbeaten is True and tag.clears == 0

    def test_the_loaded_local_application_rows_clear_nothing(
        self, store: Scratch
    ) -> None:
        """§3: the pre-site games are rated by nothing, counted by no leaderboard and
        clear no tag. They were played with undo on hand and predate the modes."""
        maintainer = store.user("maintainer")
        row = store.play(
            ALPHA_HASH, maintainer, TOP, result=1.0, source=m.LOCAL_APP
        )

        assert row.source == m.LOCAL_APP
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True

    def test_a_game_played_out_to_the_engines_own_verdict_is_read_the_same_way(
        self, store: Scratch
    ) -> None:
        """One end-to-end game, so the tag is not only ever shown stated results.

        `alpha` stops after one move each, which the engine calls a draw, and a draw
        clears nothing. The point is that the row it leaves behind -- the mode, the *top*
        bit, the group, the result -- is the one the tag reads.
        """
        ada = store.user("ada")
        row = m.start(store.connection, ALPHA_HASH, m.LEADERBOARD, ["human", TOP], ada)
        snapshot = m.session(store.connection, row.id).snapshot()
        m.play(store.connection, row.id, snapshot["moves"][0]["move"])
        finished = m.play_agent(store.connection, row.id)
        store.connection.commit()

        assert finished["outcome"] is not None, "the two-ply game did not end"
        played = m.row(store.connection, row.id)
        assert played.top and played.top_players == (TOP,)
        assert played.result == 0.5
        assert unbeaten.tag(store.connection, ALPHA_HASH).unbeaten is True


# -------------------------------------------------------------- one person's record


class TestTheRecord:
    def test_an_untried_game_says_so(self, store: Scratch) -> None:
        ada = store.user("ada")
        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.state == unbeaten.UNTRIED and found.played == 0

    def test_a_loss_is_tried(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=0.0)
        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.state == unbeaten.TRIED
        assert (found.played, found.won, found.drawn, found.lost) == (1, 0, 0, 1)

    def test_a_win_over_random_is_won_and_says_which_rung(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, BOTTOM, result=1.0)
        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.state == unbeaten.WON
        assert found.best is not None and found.best.agent == BOTTOM

    def test_the_rung_is_the_highest_beaten_at_the_rating_it_had_then(
        self, store: Scratch
    ) -> None:
        """§5's three: the highest-rated computer player beaten, its rating at the time,
        and the seat won from. All three are on the match row, so a later result moving
        that computer player cannot rewrite the record."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, BOTTOM, result=1.0)
        store.play(ALPHA_HASH, ada, MIDDLE, result=1.0, seat=1)
        before = r.load(store.connection, ALPHA_HASH)[r.computer(MIDDLE)].rating
        store.lift(ALPHA_HASH, MIDDLE, to=100.0)

        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.best is not None
        assert found.best.agent == MIDDLE
        assert found.best.seat == 1
        assert found.best.rating is not None
        assert found.best.rating > before, "the rung read the table rather than the row"

    def test_beating_the_top_is_its_own_state(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.state == unbeaten.BEAT_THE_TOP and found.beat_the_top

    def test_half_a_tied_top_is_won_and_not_beat_the_top(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(BETA_HASH, ada, TOP, result=1.0)
        found = unbeaten.record(store.connection, BETA_HASH, ada)
        assert found.state == unbeaten.WON

    def test_the_three_counts_are_over_the_rated_games(self, store: Scratch) -> None:
        """§5's profile: tried, won and beat the top, each as *n of N*. `gamma` has no
        ladder, so it is in no denominator -- a person cannot try what is not offered."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        store.play(BETA_HASH, ada, BOTTOM, result=0.0)

        counts = unbeaten.counts(store.connection, ada)
        assert counts.rated == 2
        assert (counts.tried, counts.won, counts.top) == (2, 1, 1)

    def test_another_persons_games_are_not_on_this_record(self, store: Scratch) -> None:
        ada, bob = store.user("ada"), store.user("bob")
        store.play(ALPHA_HASH, bob, TOP, result=1.0)
        found = unbeaten.record(store.connection, ALPHA_HASH, ada)
        assert found.state == unbeaten.UNTRIED

    def test_a_loaded_local_application_row_is_on_the_profile_and_in_no_count(
        self, store: Scratch
    ) -> None:
        """§3 keeps them for the anchor's offline reading and counts them nowhere else."""
        maintainer = store.user("maintainer")
        store.play(ALPHA_HASH, maintainer, TOP, result=1.0, source=m.LOCAL_APP)

        counts = unbeaten.counts(store.connection, maintainer)
        assert (counts.tried, counts.won, counts.top) == (0, 0, 0)
        assert m.human_games(store.connection, ALPHA_HASH), "the anchor lost the row"


# ------------------------------------------------------------------------ the listing


def _cards(store: Scratch) -> list[dict[str, Any]]:
    """The three games as the games page sees them, with the store's columns on them."""
    found = browse.columns(store.connection, [ALPHA_HASH, BETA_HASH, GAMMA_HASH])
    summaries = [
        {"hash": ALPHA_HASH, "name": "alpha", "index": 0.2},
        {"hash": BETA_HASH, "name": "beta", "index": 0.8},
        {"hash": GAMMA_HASH, "name": "gamma", "index": None},
    ]
    return browse.decorate(summaries, found)


def _names(cards: list[dict[str, Any]]) -> list[str]:
    return [str(card["name"]) for card in cards]


class TestTheListing:
    def test_with_no_sort_the_order_is_the_librarys_own(self, store: Scratch) -> None:
        """The default order is unchanged, which is the one thing the three keys must not
        quietly take away."""
        assert _names(browse.order(_cards(store))) == ["alpha", "beta", "gamma"]

    def test_it_orders_by_stars(self, store: Scratch) -> None:
        ada = store.user("ada")
        votes.cast(store.connection, ada, ALPHA_HASH, 2)
        votes.cast(store.connection, ada, BETA_HASH, 5)

        ordered = browse.order(_cards(store), browse.STARS)

        assert _names(ordered) == ["beta", "alpha", "gamma"]
        assert ordered[0]["stars"] == 5.0 and ordered[0]["votes"] == 1
        assert ordered[-1]["stars"] is None, "an unvoted game was scored zero"

    def test_it_orders_by_human_games(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(BETA_HASH, ada, BOTTOM, result=1.0)
        store.play(BETA_HASH, ada, RIVAL, result=0.0)
        store.play(ALPHA_HASH, ada, MIDDLE, result=0.0)

        ordered = browse.order(_cards(store), browse.HUMAN_GAMES)

        assert _names(ordered) == ["beta", "alpha", "gamma"]
        assert [card["humanGames"] for card in ordered] == [2, 1, 0]

    def test_human_games_count_no_sandbox_and_no_loaded_rows(
        self, store: Scratch
    ) -> None:
        """§4 and §3 between them: sandbox counts for nothing, and the maintainer's
        pre-site games are counted by no leaderboard."""
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0, mode=m.SANDBOX)
        store.play(ALPHA_HASH, ada, TOP, result=1.0, source=m.LOCAL_APP)
        store.play(ALPHA_HASH, ada, MIDDLE, result=None)

        found = browse.columns(store.connection, [ALPHA_HASH])[ALPHA_HASH]
        assert found.human_games == 0

    def test_it_orders_by_index(self, store: Scratch) -> None:
        """The card's own number, which is `canonical.json`'s. A game with no current
        measurement has no index and sorts last rather than first."""
        ordered = browse.order(_cards(store), browse.INDEX)
        assert _names(ordered) == ["beta", "alpha", "gamma"]

    def test_it_filters_on_unbeaten(self, store: Scratch) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)

        kept = browse.order(_cards(store), keep=browse.UNBEATEN)

        assert _names(kept) == ["beta"], "an unrated or a beaten game stayed in the list"

    def test_an_unrated_game_is_in_neither_half_of_the_filter(
        self, store: Scratch
    ) -> None:
        card = next(c for c in _cards(store) if c["name"] == "gamma")
        assert card["rated"] is False
        assert card["unbeaten"] is None

    def test_an_unknown_sort_or_filter_is_refused_by_name(self, store: Scratch) -> None:
        """A listing silently in the wrong order is a listing that lies about its own
        ranking, so an unrecognised key is a refusal rather than a shrug."""
        for bad in ("elo", "human-games"):
            with pytest.raises(browse.BrowseError) as error:
                browse.order(_cards(store), bad)
            assert "human_games" in str(error.value)
        with pytest.raises(browse.BrowseError):
            browse.order(_cards(store), keep="beaten")

    def test_a_game_the_store_has_never_seen_keeps_its_card(
        self, store: Scratch
    ) -> None:
        found = browse.columns(store.connection, [ALPHA_HASH])
        cards = browse.decorate([{"hash": "0" * 12, "name": "stranger"}], found)
        assert cards == [{"hash": "0" * 12, "name": "stranger"}]


# ------------------------------------------------------------------------- the routes


@pytest.fixture
def app(store: Scratch, tmp_path: Path) -> SiteApp:
    """The site over the scratch store, with the three games on disk as its library."""
    rulesets = tmp_path / "rulesets"
    rulesets.mkdir()
    for spec in (ALPHA, BETA, GAMMA):
        (rulesets / f"{spec['name']}.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
    runs = tmp_path / "runs"
    runs.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    return SiteApp(rulesets, runs, tmp_path / "heuristics", archive, store.open)


class TestTheRoutes:
    def test_the_listing_carries_the_stores_columns_and_says_what_it_offers(
        self, app: SiteApp, store: Scratch
    ) -> None:
        ada = store.user("ada")
        votes.cast(store.connection, ada, ALPHA_HASH, 4)
        store.connection.commit()

        payload = app.get("/api/library")

        cards = {card["name"]: card for card in payload["games"]}
        assert cards["alpha"]["stars"] == 4.0 and cards["alpha"]["unbeaten"] is True
        assert cards["gamma"]["rated"] is False and cards["gamma"]["unbeaten"] is None
        # *newest on trial* only while a game is on trial (L4); nothing here is.
        assert [entry["sort"] for entry in payload["site"]["sorts"]] == [
            browse.STARS, browse.HUMAN_GAMES
        ]
        assert payload["site"]["filters"][0]["filter"] == browse.UNBEATEN

    def test_the_listing_takes_a_sort_and_a_filter_from_the_query(
        self, app: SiteApp, store: Scratch
    ) -> None:
        ada = store.user("ada")
        votes.cast(store.connection, ada, BETA_HASH, 5)
        store.play(ALPHA_HASH, ada, TOP, result=1.0)

        by_stars = app.get("/api/library", {"sort": browse.STARS})
        unbeaten_only = app.get("/api/library", {"filter": browse.UNBEATEN})

        assert by_stars["games"][0]["name"] == "beta"
        assert [card["name"] for card in unbeaten_only["games"]] == ["beta"]

    def test_the_default_listing_is_the_librarys_own_order(self, app: SiteApp) -> None:
        plain = [card["name"] for card in app.get("/api/library")["games"]]
        assert plain == sorted(plain), "the default order moved"

    def test_a_bad_sort_comes_back_as_a_refusal(self, app: SiteApp) -> None:
        with pytest.raises(ValueError, match="sort is one of"):
            app.get("/api/library", {"sort": "elo"})

    def test_the_game_page_carries_the_tag_and_the_target(
        self, app: SiteApp, store: Scratch
    ) -> None:
        payload = app.get(f"/api/game/{ALPHA_HASH}")
        standing = payload["standing"]
        assert standing["rated"] and standing["unbeaten"] is True
        assert standing["top"] == [TOP]
        assert "yourRecord" not in standing, "a signed-out visitor got somebody's record"

    def test_the_game_page_carries_this_persons_record_once_somebody_is_signed_in(
        self, app: SiteApp, store: Scratch
    ) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, MIDDLE, result=1.0)
        app.viewer = ada

        standing = app.get(f"/api/game/{ALPHA_HASH}")["standing"]

        assert standing["yourRecord"]["state"] == unbeaten.WON
        assert standing["yourRecord"]["rung"]["agent"] == MIDDLE

    def test_a_vote_is_cast_through_the_route(
        self, app: SiteApp, store: Scratch
    ) -> None:
        app.viewer = store.user("ada")
        answer = app.post("/api/vote", {"hash": ALPHA_HASH, "stars": 5})
        assert answer == {
            "hash": ALPHA_HASH, "stars": 5.0, "votes": 1, "tags": [], "tagCounts": []
        }

    def test_voting_needs_somebody_to_be_signed_in(self, app: SiteApp) -> None:
        with pytest.raises(ValueError, match="sign in"):
            app.post("/api/vote", {"hash": ALPHA_HASH, "stars": 5})

    def test_the_profile_carries_the_three_counts(
        self, app: SiteApp, store: Scratch
    ) -> None:
        ada = store.user("ada")
        store.play(ALPHA_HASH, ada, TOP, result=1.0)
        app.viewer = ada

        profile = app.get("/api/profile")

        # `highest` is S2's count, which A2 added beside the three. Nought here: one
        # match leaves a rating provisional, and the table ranks no provisional person.
        assert profile["counts"] == {
            "rated": 2, "tried": 1, "won": 1, "top": 1, "highest": 0
        }
        alpha = next(row for row in profile["games"] if row["hash"] == ALPHA_HASH)
        assert alpha["state"] == unbeaten.BEAT_THE_TOP and alpha["unbeaten"] is False
