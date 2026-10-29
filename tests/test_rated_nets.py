"""A kept learned net joins its game's measuring roster (Q502h).

The user's answer of 2026-10-03 (`docs/archive/decisions.md`, *When a better computer player
arrives*): each kept generation reaches the measuring set at once and its games are measured
again. `docs/reference/general-player.md` §7 has the rule; what is held here:

- **Which net is rated** (`nets.rated_net`): accepted and kept, or accepted and confirmed on
  fresh seeds where no generation line runs; never one the plan read and did not keep; the
  newest generation of those. On synthetic stores, then pinned on the committed ones, and
  held to the site's own pick (`web.opponents.learned_for`) on every library game.
- **What the roster becomes** (`corpus.canonical_agents`): every searching rung plays the net
  at the same budget, `random` and `greedy` unchanged; a game with no rated net keeps
  `CANONICAL_AGENTS` to the letter.
- **What the score rests on** (`corpus.current_basis`): the net ladder and `Basis.net` naming
  the bytes, so a kept generation stales its game's score and nothing else's.
- **What a measurement plays**: the net ladder (the runner stubbed).
- **What the trust line reads** (`trust` E-N): once the score plays the rated net, the
  earlier generations it was chosen over question nothing; a newer one still does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC

from chess_variant_sandbox import corpus, trust
from chess_variant_sandbox.agents.nets import (
    ACCEPTED,
    INPUTS_VERSION,
    REFUSED,
    STORE_DIR,
    random_net,
    rated_net,
)
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.web import opponents

ROOT = Path(__file__).resolve().parent.parent
DIGEST = spec_hash(SPEC)
BUDGETS = (100, 200, 400, 800, 1600, 6400)

# The committed stores this task put in the roster, game by game (2026-10-14; Q969g's
# generation 3c moved intransitive's and gardner's on 2026-10-20).
ROLLED_OUT = {
    "29d5213b02b2": "b14f1d190426",  # intransitive: generation 3c, kept (archive/plan-teacher §9.20)
    "82a4788c02f0": "c30820cdf2de",  # rps-v2: generation 1, kept (pilot-plan §6.6)
    "66b001f1b790": "1851f0c2e373",  # racingkings: generation 1, kept (pilot-plan §6.7)
    "975dde173f01": "ae241123e674",  # gardner: 3c's guard child, accepted and confirmed (§9.20)
    "71912ea8ba7f": "db5ef1bc938e",  # kingofthehill: generation 0, accepted and confirmed
    # L2's batch 2 (plan-teacher §10.11, 2026-10-24): generation 0, accepted and confirmed;
    # LG's child of dobutsu was accepted and confirmed but not kept (G-P2), so gen 0 stays.
    "a2f0c65ceedf": "db5ef1bc938e",  # dobutsu
    # LG (plan-teacher §10.12-§10.13, 2026-10-29): generation 1, kept (twelve games; three of
    # them, antichess, circe and rifle, on a refused generation-0 parent).
    "0be72e61ef24": "c68907df8d4a",  # marchfield
    "2261f8203948": "b0b1d779058c",  # rps annihilation (rps-v1)
    "3c639d277808": "fdaace93f9d5",  # demotion chess (hierarchy-fall)
    "3ec1560124c3": "0a83806cb796",  # antichess
    "54a0a193da0f": "9e725621900e",  # circe
    "57667d7861ad": "fb172640322f",  # amazon ladder (hierarchy-extended)
    "a1805ac1adc1": "a3dd06ddf2ff",  # kyoto
    "add543cc28c3": "ab3f11451433",  # rps repaint (rps-v4)
    "bdf2801ade24": "9946eb49d8ae",  # rps race (rps-v3)
    "f7a789809758": "2372dbe1cacb",  # rifle
    "fb8a67625131": "ae8c05a4915c",  # jeson mor
    "ffc7b604bea0": "84055e9929a5",  # reckoning
}


def _store(
    root: Path,
    procedure: str,
    *,
    digest: str = DIGEST,
    verdict: str = ACCEPTED,
    kept: bool | None = None,
    confirmed: bool | None = True,
    parent: str = "",
    score: float = 0.6,
    weights: bytes | None = None,
) -> bytes:
    """A store pair on disk: random weights, and a JSON the way the trainer and plan write it."""
    root.mkdir(parents=True, exist_ok=True)
    blob = random_net(10, seed=int(procedure, 16) % 1000) if weights is None else weights
    (root / f"{digest}-{procedure}.net").write_bytes(blob)
    data: dict[str, Any] = {
        "procedure": procedure, "ruleset_hash": digest, "inputs_version": INPUTS_VERSION,
        "verdict": verdict, "weights_sha256": hashlib.sha256(blob).hexdigest(),
        "acceptance": {"candidate": f"minimax:net={procedure},nodes=6400", "score": score,
                       "error": 0.02, "games": 400, "accepted": verdict == ACCEPTED},
    }
    if confirmed is not None:
        data["confirmation"] = {"score": score, "error": 0.02, "games": 400,
                                "accepted": confirmed}
    if kept is not None:
        data["kept"] = kept
    if parent:
        data["lineage"] = {"parent": parent}
    (root / f"{digest}-{procedure}.json").write_text(json.dumps(data), encoding="utf-8")
    return blob


# ------------------------------------------------------------------ which net is rated


class TestWhichNetIsRated:
    def test_none_without_a_store(self, tmp_path: Path) -> None:
        assert rated_net(DIGEST, tmp_path) == ""

    def test_an_accepted_and_confirmed_net_is_rated(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001")
        assert rated_net(DIGEST, tmp_path) == "000000000001"

    def test_a_refused_or_unconfirmed_net_is_not(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001", verdict=REFUSED)
        _store(tmp_path, "000000000002", confirmed=False)
        _store(tmp_path, "000000000003", confirmed=None)
        assert rated_net(DIGEST, tmp_path) == ""

    def test_a_kept_generation_counts_as_confirmed(self, tmp_path: Path) -> None:
        """*Intransitive*'s carried-over generations record their G-A2 on the old hash."""
        _store(tmp_path, "000000000001", kept=True, confirmed=None)
        assert rated_net(DIGEST, tmp_path) == "000000000001"

    def test_one_the_plan_did_not_keep_never_plays(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001", kept=True)
        _store(tmp_path, "000000000002", kept=False, parent="000000000001", score=0.9)
        assert rated_net(DIGEST, tmp_path) == "000000000001"

    def test_the_newest_kept_generation_wins_over_an_accepted_child(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001", kept=True)
        _store(tmp_path, "000000000002", kept=True, parent="000000000001")
        # Accepted, confirmed, no verdict written: not kept, so not the line's player.
        _store(tmp_path, "000000000003", parent="000000000002", score=0.95)
        assert rated_net(DIGEST, tmp_path) == "000000000002"

    def test_without_a_line_the_newest_accepted_generation_plays(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001", score=0.7)
        _store(tmp_path, "000000000002", parent="000000000001", score=0.65)
        assert rated_net(DIGEST, tmp_path) == "000000000002"

    def test_a_missing_or_altered_weights_file_is_not_rated(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001")
        (tmp_path / f"{DIGEST}-000000000001.net").unlink()
        _store(tmp_path, "000000000002")
        (tmp_path / f"{DIGEST}-000000000002.net").write_bytes(random_net(10, seed=99))
        assert rated_net(DIGEST, tmp_path) == ""

    def test_another_games_store_is_not_this_games(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001", digest="f" * 12)
        assert rated_net(DIGEST, tmp_path) == ""


class TestTheCommittedStores:
    def test_the_five_games_and_their_nets(self) -> None:
        for digest, procedure in ROLLED_OUT.items():
            assert rated_net(digest, ROOT / STORE_DIR) == procedure, digest

    def test_no_other_library_game_has_one(self) -> None:
        """Chess's nets are all refused; every other game has none."""
        seen = set()
        for directory in ("rulesets", "archive"):
            for path in sorted((ROOT / directory).rglob("*.json")):
                try:
                    digest = spec_hash(load_spec(path))
                except Exception:  # noqa: BLE001 - not a spec (an archive's index)
                    continue
                seen.add(digest)
                assert rated_net(digest, ROOT / STORE_DIR) == ROLLED_OUT.get(digest, ""), path
        assert set(ROLLED_OUT) <= seen

    def test_the_rated_net_is_the_sites_learned_player(self) -> None:
        """One rule read two ways: the measuring roster seats what the site offers, less any
        net not confirmed on a second seed set (none, among the library's)."""
        for digest, procedure in ROLLED_OUT.items():
            (offered,) = opponents.learned_for(digest, ROOT / STORE_DIR)
            assert offered.agent == f"minimax:net={procedure},nodes=6400", digest


# ------------------------------------------------------------------ the roster and its basis


class TestTheRoster:
    def test_a_game_without_a_rated_net_keeps_the_canonical_roster(self, tmp_path: Path) -> None:
        assert corpus.canonical_agents(DIGEST, tmp_path) == corpus.CANONICAL_AGENTS

    def test_a_rated_net_plays_every_searching_rung_at_its_budget(self, tmp_path: Path) -> None:
        _store(tmp_path, "000000000001")
        roster = corpus.canonical_agents(DIGEST, tmp_path)
        assert roster == ("random", "greedy") + tuple(
            f"minimax:net=000000000001,nodes={budget}" for budget in BUDGETS
        )
        # The shape a score is compared in does not move: eight players, the same budgets.
        assert len(roster) == len(corpus.CANONICAL_AGENTS)
        assert [a.rpartition("nodes=")[2] for a in roster[2:]] == [
            a.rpartition("nodes=")[2] for a in corpus.CANONICAL_AGENTS[2:]
        ]

    def test_the_basis_names_the_net(self, tmp_path: Path, monkeypatch) -> None:
        before = corpus.current_basis(DIGEST, spec=SPEC)
        blob = _store(tmp_path, "000000000001")
        monkeypatch.setattr(corpus, "NETS_DIR", tmp_path)
        # `Basis.net` reads the bytes the way a seat does: through the registry, which
        # `rated_net` filled from this directory.
        now = corpus.current_basis(DIGEST, spec=SPEC)
        assert now.agents == corpus.canonical_agents(DIGEST, tmp_path)
        assert now.net == "000000000001=net:" + hashlib.sha256(blob).hexdigest()[:12]
        assert now.players and now.players != before.players
        staleness = before.stale_against(now)
        assert staleness.needs == "replay"
        assert any("different roster" in reason for reason in staleness.reasons)

    def test_the_committed_games_read_their_net_ladders(self) -> None:
        for digest, procedure in ROLLED_OUT.items():
            basis = corpus.current_basis(digest)
            assert basis.agents[-1] == f"minimax:net={procedure},nodes=6400"
            assert basis.net.startswith(f"{procedure}=net:")
        chess = corpus.current_basis(spec_hash(load_spec(ROOT / "rulesets" / "chess.json")))
        assert chess.agents == corpus.CANONICAL_AGENTS and chess.net == ""


class TestAMeasurementPlaysTheNetLadder:
    def test_it_plays_the_net_roster(self, tmp_path: Path, monkeypatch) -> None:
        runs = tmp_path / "runs"
        nets = tmp_path / "nets"
        _store(nets, "000000000001")
        monkeypatch.setattr(corpus, "NETS_DIR", nets)
        monkeypatch.setattr(corpus, "EXPECTED_LIBRARY_SEEDS", corpus.ONE_SEED)

        played: list[tuple[str, ...]] = []

        class _Tournament:
            records: list[Any] = []

            class ratings:  # noqa: N801 - stands in for the runner's attribute
                standings: list[Any] = []

            def save(self, out: Path) -> None:
                out.mkdir(parents=True, exist_ok=True)

        def play(spec: Any, config: Any, progress: bool = False) -> _Tournament:
            played.append(tuple(config.agents))
            return _Tournament()

        def analyse(spec: Any, records: Any, seed: int = 0, workers: int = 0, **_: Any) -> Any:
            from chess_variant_sandbox.tournament.analysis import CorpusAnalysis

            return CorpusAnalysis()

        def refuse(*_: Any, **__: Any) -> Any:
            from chess_variant_sandbox.metrics.scoring import WeightsError

            raise WeightsError("stand-in: nothing to score")

        monkeypatch.setattr(corpus, "run_tournament", play)
        monkeypatch.setattr(corpus, "analyse_corpus", analyse)
        monkeypatch.setattr(corpus, "score_ruleset", refuse)

        candidate = corpus.Candidate(
            hash=DIGEST, name="lancers", spec=SPEC, stage="committed",
            path=tmp_path / "lancers.json",
        )
        stored = corpus.measure(candidate, runs_dir=runs, workers=1)
        assert played == [corpus.canonical_agents(DIGEST, nets)]
        assert stored.stored["agents"] == list(corpus.canonical_agents(DIGEST, nets))
        assert stored.basis is not None and stored.basis.net.startswith("000000000001=net:")
        assert stored.current


# ------------------------------------------------------------------ the trust line


class TestTheTrustLine:
    def _read(self, runs: Path, nets: Path, played: str) -> trust.Item:
        basis = corpus.current_basis(DIGEST, spec=SPEC).as_dict()
        basis["net"] = played
        out = corpus.measurement_dir(DIGEST, runs)
        out.mkdir(parents=True, exist_ok=True)
        seeds = corpus.LIBRARY_SEEDS
        (out / "canonical.json").write_text(json.dumps({
            "hash": DIGEST, "name": SPEC["name"], "agents": list(basis["agents"]),
            "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds),
            "games": 1024, "index": 0.7, "totals_used": 4, "sd": 0.02, "viable": True,
            "violations": [], "weights_version": basis["weights_version"], "families": {},
            "standings": [],
            "runs": [{"seed": seed, "index": 0.7, "sd": 0.02, "viable": True,
                      "violations": [], "unreadable": [], "origin": "played"}
                     for seed in seeds],
            "basis": basis,
        }), encoding="utf-8")
        return trust.read(DIGEST, runs, SPEC, None, nets_dir=nets).item(trust.NET)

    def test_earlier_generations_question_nothing_once_the_score_plays_the_rated_one(
        self, tmp_path: Path
    ) -> None:
        nets = tmp_path / "nets"
        _store(nets, "000000000001", kept=True)
        child = _store(nets, "000000000002", kept=True, parent="000000000001")
        played = "000000000002=net:" + hashlib.sha256(child).hexdigest()[:12]
        item = self._read(tmp_path / "runs", nets, played)
        assert (item.reading, item.sentence) == (trust.SILENT, "net.in-the-score")

    def test_a_newer_kept_generation_still_questions(self, tmp_path: Path) -> None:
        nets = tmp_path / "nets"
        parent = _store(nets, "000000000001", kept=True)
        _store(nets, "000000000002", kept=True, parent="000000000001")
        played = "000000000001=net:" + hashlib.sha256(parent).hexdigest()[:12]
        item = self._read(tmp_path / "runs", nets, played)
        assert (item.reading, item.sentence) == (trust.FINDS_MORE, "net.beats-ours")
        assert item.detail["procedure"] == "000000000002"

    def test_before_the_roll_out_the_accepted_net_questions(self, tmp_path: Path) -> None:
        nets = tmp_path / "nets"
        _store(nets, "000000000001", kept=True)
        item = self._read(tmp_path / "runs", nets, "")
        assert item.reading == trust.FINDS_MORE

    def test_the_net_in_the_score_is_said_before_a_refused_child(self, tmp_path: Path) -> None:
        nets = tmp_path / "nets"
        blob = _store(nets, "000000000002")
        _store(nets, "000000000001", verdict=REFUSED, parent="000000000002")
        played = "000000000002=net:" + hashlib.sha256(blob).hexdigest()[:12]
        item = self._read(tmp_path / "runs", nets, played)
        assert item.sentence == "net.in-the-score"


@pytest.fixture(autouse=True)
def _forget_registered_nets() -> Any:
    """`load_net` registers what it reads; a synthetic store must not outlive its test."""
    from chess_variant_sandbox.agents import nets

    held = dict(nets._NETS)
    yield
    nets._NETS.clear()
    nets._NETS.update(held)
