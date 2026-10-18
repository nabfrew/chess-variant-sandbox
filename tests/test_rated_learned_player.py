"""Q913: the site's learned player joins its game's one leaderboard where its net is rated.

`docs/archive/plan-leaderboard.md` §2.1 (one table per game, computer players and people together)
and `docs/plan-teacher.md` §9.7 (*the picker ladders*). Q502h put each game's rated net
(`nets.rated_net`: kept, or accepted and confirmed on fresh seeds) in its measuring roster;
here the same net is the learned player the picker offers, and it is rated:

- **which player**: on a game with a rated net the learned player is exactly that net, at
  its acceptance duel's budget, named *the learned player*, not practice only, and in the
  roster the ladder rates (`opponents.agents_of`), so `deploy/ladders_check.py` owes its
  ladder a row until it is topped up;
- **where it is not**: every other game's roster is unchanged, and a net that is not the
  game's rated one (accepted and not confirmed) stays *the learned player (experimental)*,
  practice only (Q683);
- **on the site**: a ladder seats it (`rating.seeding` with its game), the leaderboard
  counts its row, and a rated game against it starts (`test_site_matches.py`).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_words
from test_rated_nets import ROLLED_OUT

from chess_variant_sandbox.agents.nets import (
    ACCEPTED,
    STORE_DIR,
    Layout,
    NetRecord,
    random_net,
    rated_net,
    save_net,
    unregister_net,
)
from chess_variant_sandbox.site import leaderboard as board
from chess_variant_sandbox.site import rating
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import opponents
from chess_variant_sandbox.web.library import engine_for

ROOT = Path(__file__).resolve().parent.parent
NETS = ROOT / STORE_DIR


def _library() -> dict[str, tuple[Path, dict[str, Any]]]:
    """Every committed and archived spec, by hash, as the site's library lists them."""
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for directory in ("archive", "rulesets"):
        for path in sorted((ROOT / directory).glob("*.json")):
            try:
                spec = load_spec(path)
            except Exception:  # noqa: BLE001 - not a spec (an archive's index)
                continue
            found[spec_hash(spec)] = (path, spec)
    return found


def _learned(roster: list[opponents.Opponent]) -> list[opponents.Opponent]:
    return [entry for entry in roster if entry.store == "learned"]


# ------------------------------------------------------------------ the five games


class TestTheFiveGames:
    def test_the_learned_player_is_the_rated_net_and_is_rated(self) -> None:
        library = _library()
        for digest, procedure in ROLLED_OUT.items():
            _path, spec = library[digest]
            assert rated_net(digest, NETS) == procedure, digest
            roster = opponents.for_game(digest, spec=spec, nets=NETS)
            (entry,) = _learned(roster)
            # The same net the measuring roster plays, at the budget the site plays it.
            assert entry.agent == f"minimax:net={procedure},nodes=6400", digest
            assert not entry.practice_only and entry.as_dict()["practiceOnly"] is False
            assert entry.name == opponents.RATED_LEARNED_NAME
            assert entry.about == "looks at 6,400 positions before each move"
            assert entry.agent in opponents.agents_of(roster), digest
            assert not opponents.never_rated(entry.agent, digest, NETS), digest
            for words in (entry.label, entry.plan, entry.group):
                assert not site_words.hits(words), words
            assert "rated like the other computer players" in entry.plan

    def test_its_agent_string_is_the_measuring_rosters_top_rung(self) -> None:
        """The player the site rates is the player the score is measured with at 6,400
        nodes (`corpus.canonical_agents`), one identity, one string."""
        from chess_variant_sandbox import corpus

        for digest, procedure in ROLLED_OUT.items():
            (entry,) = opponents.learned_for(digest, NETS)
            assert entry.agent == corpus.canonical_agents(digest, NETS)[-1]
            assert entry.agent == f"minimax:net={procedure},nodes=6400"

    def test_another_games_net_is_never_rated_here(self) -> None:
        """A procedure names a net on one game only: the same string on a game whose rated
        net it is not stays practice only."""
        kingofthehill, intransitive = "71912ea8ba7f", "29d5213b02b2"
        agent = f"minimax:net={ROLLED_OUT[kingofthehill]},nodes=6400"
        assert not opponents.never_rated(agent, kingofthehill, NETS)
        assert opponents.never_rated(agent, intransitive, NETS)


class TestEveryOtherGame:
    def test_no_other_library_game_offers_a_rated_net(self) -> None:
        """Games without a rated net are as they were: whatever learned player they offer
        (`learned_for`, the one way a net player enters a roster) is practice only, so the
        roster the ladder rates holds no net player."""
        for digest, (path, _spec) in _library().items():
            if digest in ROLLED_OUT:
                continue
            offered = opponents.learned_for(digest, NETS)
            assert all(entry.practice_only for entry in offered), path
            assert opponents.agents_of(offered) == [], path


# ------------------------------------------------------------------ the rule, on synthetic stores


@pytest.fixture
def dull() -> Iterator[dict[str, Any]]:
    spec = load_spec(ROOT / "rulesets" / "dull.json")
    yield spec
    for procedure in ("aaaaaaaaaaa1", "aaaaaaaaaaa2"):
        unregister_net(procedure, spec_hash(spec))


def _store(root: Path, spec: dict[str, Any], procedure: str, *, confirmed: bool,
           parent: str = "", score: float = 0.7) -> None:
    """An accepted net for `spec` at 400 nodes, with or without a passed G-A2."""
    digest = spec_hash(spec)
    weights = random_net(Layout.of(engine_for(spec).spec).n_inputs,
                         seed=int(procedure, 16) % 1000, accumulator=8, hidden1=4, hidden2=4)
    meta = save_net(NetRecord(procedure, digest, ACCEPTED, weights), root)
    data = json.loads(meta.read_text("utf-8"))
    duel = {"score": score, "games": 40, "accepted": True}
    data["acceptance"] = {**duel, "candidate": f"minimax:net={procedure},nodes=400"}
    if confirmed:
        data["confirmation"] = duel
    if parent:
        data["lineage"] = {"parent": parent}
    meta.write_text(json.dumps(data), "utf-8")


class TestTheRule:
    def test_an_unconfirmed_net_is_the_experiment_and_a_confirmed_one_is_rated(
        self, tmp_path: Path, dull: dict[str, Any]
    ) -> None:
        digest = spec_hash(dull)
        _store(tmp_path, dull, "aaaaaaaaaaa1", confirmed=False)
        (entry,) = opponents.learned_for(digest, tmp_path, dull)
        assert entry.practice_only and entry.name == opponents.LEARNED_NAME
        assert opponents.never_rated(entry.agent, digest, tmp_path)
        _store(tmp_path, dull, "aaaaaaaaaaa1", confirmed=True)
        (entry,) = opponents.learned_for(digest, tmp_path, dull)
        assert not entry.practice_only and entry.name == opponents.RATED_LEARNED_NAME
        assert not opponents.never_rated(entry.agent, digest, tmp_path)

    def test_the_rated_net_is_offered_over_a_newer_unconfirmed_child(
        self, tmp_path: Path, dull: dict[str, Any]
    ) -> None:
        """Without the rule the picker would offer the newest accepted net (`_kept`) as
        practice; with a rated net on the game, the picker offers that one, rated, so the
        learned player is always the net the score is measured with."""
        digest = spec_hash(dull)
        _store(tmp_path, dull, "aaaaaaaaaaa1", confirmed=True)
        _store(tmp_path, dull, "aaaaaaaaaaa2", confirmed=False, parent="aaaaaaaaaaa1",
               score=0.9)
        assert rated_net(digest, tmp_path) == "aaaaaaaaaaa1"
        (entry,) = opponents.learned_for(digest, tmp_path, dull)
        assert entry.agent == "minimax:net=aaaaaaaaaaa1,nodes=400"
        assert not entry.practice_only


# ------------------------------------------------------------------ the site's table


RATED = "minimax:net=aaaaaaaaaaa1,nodes=6400"
EARLIER = "minimax:net=aaaaaaaaaaa0,nodes=6400"
UNRATED = "minimax:net=aaaaaaaaaaa2,nodes=6400"


def _computer(agent: str, score: float, games: int = 0) -> rating.Player:
    return rating.Player(rating.computer(agent), score, 100.0, 0.0, games)


class TestTheLeaderboard:
    ROSTER = [
        *opponents.ladder(),
        opponents.Opponent(agent=RATED, label="the learned player", group="learned",
                           store="learned", name=opponents.RATED_LEARNED_NAME),
    ]

    def _rows(self, *players: rating.Player) -> list[dict[str, Any]]:
        table = {player.key: player for player in players}
        return board.rows(table, named={}, roster=self.ROSTER, viewer=None)

    def test_the_rated_learned_player_is_a_row_and_can_be_the_strongest(self) -> None:
        rows = self._rows(_computer("minimax:nodes=200000", 2300.0),
                          _computer(RATED, 2450.0), _computer("random", 800.0))
        assert [row["name"] for row in rows] == ["the learned player", "hardest", "random"]
        assert rows[0]["top"] and not rows[0]["gone"]

    def test_an_earlier_generation_people_played_keeps_its_row(self) -> None:
        """A net replaced since keeps the games people played against it, marked no longer
        offered and named by its budget; a seed nobody played, or a net never rated, is
        not a row."""
        rows = self._rows(_computer(RATED, 2450.0), _computer(EARLIER, 2400.0, games=3),
                          _computer(UNRATED, 2500.0))
        assert [(row["name"], row["gone"]) for row in rows] == [
            ("the learned player", False),
            ("the learned player at 6,400 positions", True),
        ]
        assert [row["name"] for row in self._rows(_computer(EARLIER, 2400.0))] == []

    def test_a_ladder_seats_it_only_on_its_own_game(self, monkeypatch) -> None:
        from chess_variant_sandbox.agents import nets

        monkeypatch.setattr(nets, "rated_net",
                            lambda digest, root=STORE_DIR: "aaaaaaaaaaa1" if digest == "g" else "")
        ladder = [rating.Rung(RATED, 300.0), rating.Rung(UNRATED, 320.0),
                  rating.Rung("minimax:nodes=6400", 100.0)]
        assert [key.agent for key in rating.seeds_from_ladder(ladder, game="g")] == [
            RATED, "minimax:nodes=6400"]
        assert [key.agent for key in rating.seeds_from_ladder(ladder, game="h")] == [
            "minimax:nodes=6400"]
        assert [key.agent for key in rating.seeds_from_ladder(ladder)] == ["minimax:nodes=6400"]
