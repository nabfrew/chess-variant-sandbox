"""Q683: the learned player in the opponent menu, where an accepted net is stored.

Q706: of several accepted nets the menu offers the newest *kept* generation. Q709: a
learned player whose net cannot be loaded refuses wherever it is built -- the menu, the
practice player, the schedule -- and a seat records the net that played. Q712: generation
2's real store, filed `kept: false` under the old hash, changes none of that.

Two halves. The committed net for *Intransitive* is read as the site reads it -- named
only by its ruleset hash, which is what the menu derives the entry from -- and the generic
path runs on a synthetic net over a tiny spec, so nothing about one game is assumed by
the code under test.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_words

from chess_variant_sandbox.agents.base import native_spec
from chess_variant_sandbox.agents.nets import (
    ACCEPTED,
    REFUSED,
    STORE_DIR,
    Layout,
    NetError,
    NetRecord,
    played_by,
    random_net,
    save_net,
    unregister_net,
)
from chess_variant_sandbox.site import rating
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament
from chess_variant_sandbox.web import library, opponents
from chess_variant_sandbox.web.library import engine_for
from chess_variant_sandbox.web.play import Sessions

ROOT = Path(__file__).resolve().parent.parent
# The net generation 1 accepted on Intransitive (`docs/reference/learned-evaluator.md`
# §8.18). Named here because the test is about that file; the code reads `nets/` only.
# Carried by Q696 to the game's new hash when its draw clock moved to 200 plies
# (`d93e28d09a33` before; `docs/archive/plan-teacher.md` §9.12), re-dueled there at 0.889.
INTRANSITIVE = "29d5213b02b2"
PROCEDURE = "df1148b26ff6"
AGENT = f"minimax:net={PROCEDURE},nodes=6400"
# Generation 2R (E1, `archive/plan-teacher.md` §9.13.8), generation 1's child: kept by R5-1 once
# both guards held (Q716, §9.13.12). It stays kept, and is generation 3c's parent.
KEPT_2R = "0eaad64c4586"
# Generation 3c (the chained replay, `plan-teacher.md` §9.19), 2R's child: kept by R5-1 once
# both guards held (Q969g, §9.20), so since then it is the net the picker offers.
OFFERED = "b14f1d190426"
OFFERED_AGENT = f"minimax:net={OFFERED},nodes=6400"
# Generation 2 (T5-g2, `archive/plan-teacher.md` §9.11.2): accepted and not kept (G-N failed
# on A25), filed by Q712 under the hash its verdict was read on (§9.13.6 there).
OLD_INTRANSITIVE = "d93e28d09a33"
CHILD = "70b6ece08506"
CHILD_AGENT = f"minimax:net={CHILD},nodes=6400"
CHILD_WEIGHTS = "8ec914cecdd934c216069437d163d18dcf9c0029c9a48dac8a868533aa2b07f0"


def _spec(name: str) -> dict[str, Any]:
    return load_spec(ROOT / "rulesets" / f"{name}.json")


def _game(spec: dict[str, Any]) -> library.Game:
    return library.Game(hash=spec_hash(spec), name=str(spec.get("name", "")),
                        description="", path=Path(), source="committed", spec=spec)


def _learned(roster: list[opponents.Opponent]) -> list[opponents.Opponent]:
    return [entry for entry in roster if entry.store == "learned"]


# ------------------------------------------------------------------ the real net


class TestIntransitivesNet:
    def test_the_menu_offers_it_once_at_the_duels_budget_and_rated(self) -> None:
        """Since Q913 its net is the game's rated one (Q502h), so it is rated like any
        other player and drops *experimental*; `test_rated_learned_player.py` has the rest."""
        spec = _spec("intransitive")
        assert spec_hash(spec) == INTRANSITIVE
        roster = opponents.for_game(INTRANSITIVE, spec=spec, nets=ROOT / STORE_DIR)
        (entry,) = _learned(roster)
        assert entry.agent == OFFERED_AGENT
        assert not entry.practice_only and entry.as_dict()["practiceOnly"] is False
        assert entry.name == opponents.RATED_LEARNED_NAME
        assert entry.about == "looks at 6,400 positions before each move"
        assert "rated like the other computer players" in entry.plan
        assert "practice" not in entry.plan
        assert "96%" in entry.plan and "400 games" in entry.plan
        for words in (entry.label, entry.plan, entry.group):
            assert not site_words.hits(words), words
        # Every other player is the one it was before.
        assert [e.agent for e in roster if e.store != "learned"] == [
            e.agent for e in opponents.for_game(INTRANSITIVE, spec=spec, nets=ROOT / "absent")
        ]

    def test_it_is_put_up_for_rating_as_the_games_rated_net(self) -> None:
        """Q913: rated on its own game, where it is the rated net; any other net of the
        game, and every net read without a game, stays practice only (Q683)."""
        spec = _spec("intransitive")
        roster = opponents.for_game(INTRANSITIVE, spec=spec, nets=ROOT / STORE_DIR)
        assert OFFERED_AGENT in opponents.agents_of(roster)
        assert not opponents.never_rated(OFFERED_AGENT, INTRANSITIVE, ROOT / STORE_DIR)
        assert opponents.never_rated(AGENT, INTRANSITIVE, ROOT / STORE_DIR)
        assert opponents.never_rated(OFFERED_AGENT)
        assert not opponents.never_rated("minimax:nodes=6400")
        ladder = [rating.Rung(OFFERED_AGENT, 300.0), rating.Rung(AGENT, 250.0),
                  rating.Rung("minimax:nodes=6400", 100.0)]
        seeded = rating.seeds_from_ladder(ladder)
        assert [key.agent for key in seeded] == ["minimax:nodes=6400"]
        seeded = rating.seeds_from_ladder(ladder, game=INTRANSITIVE)
        assert [key.agent for key in seeded] == [OFFERED_AGENT, "minimax:nodes=6400"]

    def test_only_a_ruleset_with_an_accepted_net_is_offered_one(self) -> None:
        """Derived from `nets/*.json`: every committed ruleset, and the entry exactly where
        an accepted store is filed under its hash."""
        accepted = {
            json.loads(path.read_text("utf-8"))["ruleset_hash"]
            for path in (ROOT / STORE_DIR).glob("*.json")
            if json.loads(path.read_text("utf-8")).get("verdict") == ACCEPTED
        }
        assert INTRANSITIVE in accepted
        for path in sorted((ROOT / "rulesets").glob("*.json")):
            digest = spec_hash(load_spec(path))
            found = opponents.learned_for(digest, ROOT / STORE_DIR)
            assert bool(found) == (digest in accepted), path.name

    def test_the_core_plays_it_with_the_net(self) -> None:
        """Through the usual path: the session asks the core's own agent, carrying the
        net's bytes, for the move."""
        spec = _spec("intransitive")
        opponents.for_game(INTRANSITIVE, spec=spec, nets=ROOT / STORE_DIR)
        weights = (ROOT / STORE_DIR / f"{INTRANSITIVE}-{OFFERED}.net").read_bytes()
        assert native_spec(OFFERED_AGENT, INTRANSITIVE)[10] == weights
        session = Sessions().start(_game(spec), (OFFERED_AGENT, "human"))
        snapshot = session.play_agent()
        assert len(snapshot["history"]) == 1 and snapshot["history"][0]["by"] == OFFERED_AGENT
        assert snapshot["strength"][0] == "looks at 6,400 positions before each move"

    def test_a_missing_net_file_is_not_offered(self, tmp_path: Path) -> None:
        shutil.copy(ROOT / STORE_DIR / f"{INTRANSITIVE}-{PROCEDURE}.json", tmp_path)
        assert opponents.learned_for(INTRANSITIVE, tmp_path, _spec("intransitive")) == []

    def test_weights_that_do_not_match_their_hash_are_not_offered(self, tmp_path: Path) -> None:
        stem = f"{INTRANSITIVE}-{PROCEDURE}"
        shutil.copy(ROOT / STORE_DIR / f"{stem}.json", tmp_path)
        weights = bytearray((ROOT / STORE_DIR / f"{stem}.net").read_bytes())
        weights[-1] ^= 1
        (tmp_path / f"{stem}.net").write_bytes(bytes(weights))
        assert opponents.learned_for(INTRANSITIVE, tmp_path, _spec("intransitive")) == []
        # And the rest of the menu is there as ever.
        # (Compared with the menu without nets, not with the ladder: since Q696 the game
        # has no accepted fit, so its menu is the ladder alone.)
        roster = opponents.for_game(INTRANSITIVE, spec=_spec("intransitive"), nets=tmp_path)
        without = opponents.for_game(INTRANSITIVE, spec=_spec("intransitive"), nets=ROOT / "absent")
        assert not _learned(roster)
        assert [e.agent for e in roster] == [e.agent for e in without]
        assert len(roster) >= len(opponents.ladder())


# ------------------------------------------------------------------ the generic path


def _store(root: Path, spec: dict[str, Any], procedure: str, *, verdict: str = ACCEPTED,
           parent: str = "", score: float = 0.75, inputs: int | None = None,
           nodes: int = 400, kept: bool | None = None) -> bytes:
    """A synthetic net for `spec`, filed as the trainer files one, with its duel."""
    digest = spec_hash(spec)
    n_inputs = inputs or Layout.of(engine_for(spec).spec).n_inputs
    weights = random_net(n_inputs, seed=int(procedure, 16) % 1000, accumulator=8,
                         hidden1=4, hidden2=4)
    meta = save_net(NetRecord(procedure, digest, verdict, weights), root)
    data = json.loads(meta.read_text("utf-8"))
    data["acceptance"] = {
        "candidate": f"minimax:net={procedure},nodes={nodes}",
        "reference": f"minimax:nodes={nodes}", "score": score, "games": 40,
        "accepted": verdict == ACCEPTED,
    }
    if parent:
        data["lineage"] = {"parent": parent}
    if kept is not None:
        data["kept"] = kept
    meta.write_text(json.dumps(data), "utf-8")
    return weights


@pytest.fixture
def tiny() -> Iterator[dict[str, Any]]:
    spec = _spec("dull")
    yield spec
    for procedure in ("aaaaaaaaaaa1", "aaaaaaaaaaa2", "aaaaaaaaaaa3"):
        unregister_net(procedure, spec_hash(spec))


class TestTheGenericPath:
    def test_an_accepted_net_is_offered_and_played(self, tmp_path: Path, tiny) -> None:
        weights = _store(tmp_path, tiny, "aaaaaaaaaaa1")
        digest = spec_hash(tiny)
        (entry,) = opponents.learned_for(digest, tmp_path, tiny)
        assert entry.agent == "minimax:net=aaaaaaaaaaa1,nodes=400"
        assert "75%" in entry.plan and "easy" in entry.plan
        # Accepted and never confirmed on a second seed set: not a rated net, so the
        # experiment of Q683, practice only.
        assert entry.practice_only and entry.name == opponents.LEARNED_NAME
        assert "not rated" in entry.plan and "practice" in entry.plan
        assert native_spec(entry.agent, digest)[10] == weights
        session = Sessions().start(_game(tiny), (entry.agent, "human"))
        assert len(session.play_agent()["history"]) == 1

    def test_the_newest_generation_is_the_one_offered(self, tmp_path: Path, tiny) -> None:
        _store(tmp_path, tiny, "aaaaaaaaaaa1", score=0.9)
        _store(tmp_path, tiny, "aaaaaaaaaaa2", parent="aaaaaaaaaaa1", score=0.6)
        (entry,) = opponents.learned_for(spec_hash(tiny), tmp_path, tiny)
        assert "aaaaaaaaaaa2" in entry.agent

    def test_a_refused_net_is_not_offered(self, tmp_path: Path, tiny) -> None:
        _store(tmp_path, tiny, "aaaaaaaaaaa1", verdict=REFUSED)
        assert opponents.learned_for(spec_hash(tiny), tmp_path, tiny) == []

    def test_a_net_of_another_layout_is_not_offered(self, tmp_path: Path, tiny) -> None:
        layout = Layout.of(engine_for(tiny).spec).n_inputs
        _store(tmp_path, tiny, "aaaaaaaaaaa3", inputs=layout + 1)
        assert opponents.learned_for(spec_hash(tiny), tmp_path, tiny) == []

    def test_no_store_at_all_is_no_entry(self, tmp_path: Path, tiny) -> None:
        assert opponents.learned_for(spec_hash(tiny), tmp_path, tiny) == []

    def test_an_accepted_child_does_not_replace_a_kept_parent(self, tmp_path: Path, tiny) -> None:
        """Q706: the picker offers the newest *kept* generation. An accepted child stored
        beside a kept parent -- T5-g2's generation 2 beside generation 1 -- is not offered,
        whether its store says `kept: false` or says nothing."""
        _store(tmp_path, tiny, "aaaaaaaaaaa1", score=0.9, kept=True)
        _store(tmp_path, tiny, "aaaaaaaaaaa2", parent="aaaaaaaaaaa1", score=0.95)
        (entry,) = opponents.learned_for(spec_hash(tiny), tmp_path, tiny)
        assert "aaaaaaaaaaa1" in entry.agent and "90%" in entry.plan
        _store(tmp_path, tiny, "aaaaaaaaaaa2", parent="aaaaaaaaaaa1", score=0.95, kept=False)
        (entry,) = opponents.learned_for(spec_hash(tiny), tmp_path, tiny)
        assert "aaaaaaaaaaa1" in entry.agent

    def test_a_kept_child_replaces_its_kept_parent(self, tmp_path: Path, tiny) -> None:
        _store(tmp_path, tiny, "aaaaaaaaaaa1", score=0.9, kept=True)
        _store(tmp_path, tiny, "aaaaaaaaaaa2", parent="aaaaaaaaaaa1", score=0.6, kept=True)
        _store(tmp_path, tiny, "aaaaaaaaaaa3", parent="aaaaaaaaaaa2", score=0.99, kept=False)
        (entry,) = opponents.learned_for(spec_hash(tiny), tmp_path, tiny)
        assert "aaaaaaaaaaa2" in entry.agent

    def test_a_net_the_plan_did_not_keep_is_never_offered(self, tmp_path: Path, tiny) -> None:
        _store(tmp_path, tiny, "aaaaaaaaaaa1", kept=False)
        assert opponents.learned_for(spec_hash(tiny), tmp_path, tiny) == []


# ----------------------------------------------- Q706 on Intransitive's own stores


class TestIntransitivesKeptNet:
    def test_its_stores_say_which_generations_were_kept(self) -> None:
        for procedure in (OFFERED, KEPT_2R, PROCEDURE, "db5ef1bc938e"):
            path = ROOT / STORE_DIR / f"{INTRANSITIVE}-{procedure}.json"
            assert json.loads(path.read_text("utf-8"))["kept"] is True, procedure

    @pytest.mark.parametrize("kept", [None, False], ids=["no-verdict", "not-kept"])
    def test_generation_two_in_nets_does_not_replace_the_kept_line(
        self, tmp_path: Path, kept: bool | None
    ) -> None:
        """The test that failed when T5-g2's `70b6ece08506` was tried in `nets/`
        (`archive/plan-teacher.md` §9.11.2). Since Q712 its real store is in the repository, under
        the old hash; here it is filed under the current one, the worst case (a store that
        claims a verdict on this game), with its `kept` verdict and without one: accepted,
        its parent generation 1, and a higher acceptance score than generation 1. The kept
        line's newest, generation 3c (kept by Q969g), is what is offered."""
        spec = _spec("intransitive")
        for path in (ROOT / STORE_DIR).glob(f"{INTRANSITIVE}-*"):
            shutil.copy(path, tmp_path)
        data = json.loads((ROOT / STORE_DIR / f"{OLD_INTRANSITIVE}-{CHILD}.json").read_text("utf-8"))
        data["ruleset_hash"] = INTRANSITIVE
        if kept is None:
            del data["kept"], data["kept_by"]
        (tmp_path / f"{INTRANSITIVE}-{CHILD}.json").write_text(json.dumps(data), "utf-8")
        shutil.copy(ROOT / STORE_DIR / f"{OLD_INTRANSITIVE}-{CHILD}.net",
                    tmp_path / f"{INTRANSITIVE}-{CHILD}.net")
        try:
            roster = opponents.for_game(INTRANSITIVE, spec=spec, nets=tmp_path)
            (entry,) = _learned(roster)
            assert entry.agent == OFFERED_AGENT
        finally:
            unregister_net(CHILD, INTRANSITIVE)


# ------------------------------------- Q712: generation 2 filed, generation 1 still offered

@pytest.fixture
def at_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Players resolve their nets from the repository's `nets/`, as a site box does."""
    monkeypatch.chdir(ROOT)
    for digest in (INTRANSITIVE, OLD_INTRANSITIVE):
        unregister_net(CHILD, digest)
    yield
    for digest in (INTRANSITIVE, OLD_INTRANSITIVE):
        unregister_net(CHILD, digest)


class TestGenerationTwosStore:
    def test_it_is_filed_whole_under_the_old_hash_and_not_kept(self) -> None:
        import hashlib

        meta = ROOT / STORE_DIR / f"{OLD_INTRANSITIVE}-{CHILD}.json"
        data = json.loads(meta.read_text("utf-8"))
        assert data["verdict"] == ACCEPTED and data["kept"] is False
        assert "G-N" in data["kept_by"] and "9.11.2" in data["kept_by"]
        assert data["ruleset_hash"] == OLD_INTRANSITIVE and data["procedure"] == CHILD
        assert data["weights_sha256"] == CHILD_WEIGHTS
        weights = (ROOT / STORE_DIR / f"{OLD_INTRANSITIVE}-{CHILD}.net").read_bytes()
        assert hashlib.sha256(weights).hexdigest() == CHILD_WEIGHTS
        assert data["lineage"]["parent"] == PROCEDURE
        assert data["acceptance"]["candidate"] == CHILD_AGENT
        assert data["confirmation"]["seed"] == 2_000_000
        # Nothing claims a verdict on the current game it never had.
        assert not list((ROOT / STORE_DIR).glob(f"{INTRANSITIVE}-{CHILD}.*"))

    def test_e1s_child_is_kept_after_its_guards(self) -> None:
        """Q713 (archive/plan-teacher §9.13.8): generation 2R, `0eaad64c4586`, passed every target
        gate and was stored whole under the current hash with no `kept` verdict; Q716
        (§9.13.12) wrote `kept: true` once G-G held on gardner and on chess (R5-1). It keeps
        it in every branch of generation 3c (§9.19.3), whose child replaced it in the picker."""
        data = json.loads((ROOT / STORE_DIR / f"{INTRANSITIVE}-{KEPT_2R}.json").read_text("utf-8"))
        assert data["verdict"] == ACCEPTED and data["kept"] is True
        assert "R5-1" in data["kept_by"] and "9.13.12" in data["kept_by"]
        assert data["lineage"]["parent"] == PROCEDURE and data["recipe"]["replay"]
        assert data["confirmation"]["seed"] == 2_000_000
        assert data["weights_sha256"].startswith("f82ee574")

    def test_generation_3c_is_kept_after_its_guards_and_offered(self) -> None:
        """Q969g (plan-teacher §9.20): generation 3c, `b14f1d190426`, 2R's chained-replay child,
        passed every target gate and G-G held on gardner and on chess (R5-1), so it is kept
        and, as the kept line's newest, what the picker offers."""
        data = json.loads((ROOT / STORE_DIR / f"{INTRANSITIVE}-{OFFERED}.json").read_text("utf-8"))
        assert data["verdict"] == ACCEPTED and data["kept"] is True
        assert "R5-1" in data["kept_by"] and "9.20" in data["kept_by"]
        assert data["lineage"]["parent"] == KEPT_2R and data["recipe"]["replay"]
        assert data["confirmation"]["seed"] == 2_000_000
        assert data["weights_sha256"].startswith("6ce71c69")
        (entry,) = opponents.learned_for(INTRANSITIVE, ROOT / STORE_DIR, _spec("intransitive"))
        assert entry.agent == OFFERED_AGENT

    def test_the_picker_offers_the_kept_line_on_both_hashes(self) -> None:
        spec = _spec("intransitive")
        (entry,) = opponents.learned_for(INTRANSITIVE, ROOT / STORE_DIR, spec)
        assert entry.agent == OFFERED_AGENT
        # Under the old hash, where generations 1 and 2 lie, the kept one is offered.
        try:
            (old,) = opponents.learned_for(OLD_INTRANSITIVE, ROOT / STORE_DIR)
            assert old.agent == AGENT
        finally:
            unregister_net(PROCEDURE, OLD_INTRANSITIVE)

    def test_no_construction_path_plays_generation_two_on_the_current_game(
        self, at_root: None
    ) -> None:
        """The practice player, the schedule and the seat record each refuse it (Q709: no
        store under this hash), and generation 1 plays through every one of them."""
        spec = _spec("intransitive")
        with pytest.raises(NetError, match="no net is stored"):
            Sessions().start(_game(spec), (CHILD_AGENT, "human"))
        config = TournamentConfig(agents=(CHILD_AGENT, "greedy"), games_per_pairing=1, workers=1)
        with pytest.raises(NetError, match=CHILD):
            run_tournament(spec, config)
        with pytest.raises(NetError, match=CHILD):
            played_by(CHILD_AGENT, INTRANSITIVE)
        # Generation 1 is what the same paths play.
        weights = (ROOT / STORE_DIR / f"{INTRANSITIVE}-{PROCEDURE}.net").read_bytes()
        assert native_spec(AGENT, INTRANSITIVE)[10] == weights
        session = Sessions().start(_game(spec), (AGENT, "human"))
        assert session.played_by[0] == played_by(AGENT, INTRANSITIVE) != ""


# --------------------------------------------- Q709: a net that cannot load refuses


@pytest.fixture
def default_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny) -> Path:
    """An empty `nets/` where a player resolves its net from, the way a site box does."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / STORE_DIR).mkdir()
    return tmp_path / STORE_DIR


def _break(store: Path, spec: dict[str, Any], procedure: str, how: str) -> str:
    """A learned player whose net, filed in `store`, cannot be loaded, and its agent string."""
    _store(store, spec, procedure)
    unregister_net(procedure, spec_hash(spec))
    stem = store / f"{spec_hash(spec)}-{procedure}"
    if how == "missing":
        stem.with_suffix(".net").unlink()
    elif how == "corrupt":
        weights = bytearray(stem.with_suffix(".net").read_bytes())
        weights[len(weights) // 2] ^= 0xFF
        stem.with_suffix(".net").write_bytes(bytes(weights))
    elif how == "unreadable":
        stem.with_suffix(".json").write_text("{ not json", "utf-8")
    elif how == "refused":
        _store(store, spec, procedure, verdict=REFUSED)
        unregister_net(procedure, spec_hash(spec))
    return f"minimax:net={procedure},nodes=400"


BROKEN = ["missing", "corrupt", "unreadable", "refused", "absent"]


class TestANetThatCannotLoad:
    """Q709: every place a net opponent is built refuses, in a sentence, rather than play
    the built-in evaluation under the learned player's name."""

    @pytest.mark.parametrize("how", BROKEN)
    def test_the_site_does_not_offer_it(self, default_store: Path, tiny, how: str) -> None:
        if how != "absent":
            _break(default_store, tiny, "aaaaaaaaaaa1", how)
        assert opponents.learned_for(spec_hash(tiny), default_store, tiny) == []

    @pytest.mark.parametrize("how", BROKEN)
    def test_the_practice_player_refuses_to_start(
        self, default_store: Path, tiny, how: str
    ) -> None:
        agent = "minimax:net=aaaaaaaaaaa1,nodes=400"
        if how != "absent":
            agent = _break(default_store, tiny, "aaaaaaaaaaa1", how)
        sessions = Sessions()
        with pytest.raises(NetError, match="built-in evaluation") as error:
            sessions.start(_game(tiny), (agent, "human"))
        assert "aaaaaaaaaaa1" in str(error.value)

    @pytest.mark.parametrize("how", BROKEN)
    def test_the_schedule_refuses_to_build(self, default_store: Path, tiny, how: str) -> None:
        agent = "minimax:net=aaaaaaaaaaa1,nodes=400"
        if how != "absent":
            agent = _break(default_store, tiny, "aaaaaaaaaaa1", how)
        config = TournamentConfig(agents=(agent, "greedy"), games_per_pairing=2, workers=1)
        with pytest.raises(NetError, match="built-in evaluation"):
            run_tournament(tiny, config)

    def test_a_net_gone_mid_game_refuses_the_next_reply(self, default_store: Path, tiny) -> None:
        """Offered and started while its net loaded; the net goes (a deploy without its
        `.net`); the next reply refuses rather than move with the built-in evaluation."""
        _store(default_store, tiny, "aaaaaaaaaaa1")
        digest = spec_hash(tiny)
        (entry,) = opponents.learned_for(digest, default_store, tiny)
        session = Sessions().start(_game(tiny), ("human", entry.agent))
        unregister_net("aaaaaaaaaaa1", digest)
        (default_store / f"{digest}-aaaaaaaaaaa1.net").unlink()
        session.play(session.snapshot()["moves"][0]["move"])
        with pytest.raises(NetError, match="built-in evaluation"):
            session.play_agent()
        assert len(session.played) == 1

    def test_the_seat_records_the_net_that_played(self, default_store: Path, tiny) -> None:
        weights = _store(default_store, tiny, "aaaaaaaaaaa1")
        (entry,) = opponents.learned_for(spec_hash(tiny), default_store, tiny)
        session = Sessions().start(_game(tiny), (entry.agent, "human"))
        digest = NetRecord("aaaaaaaaaaa1", spec_hash(tiny), ACCEPTED, weights).digest
        assert session.played_by == (f"net:{digest}", "")
        assert played_by("minimax:nodes=400", spec_hash(tiny)) == ""
        assert played_by(entry.agent, spec_hash(tiny)) == f"net:{digest}"
