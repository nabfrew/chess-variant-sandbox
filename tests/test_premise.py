"""The strength-premise check (Q502d): played, stamped beside the score, and read.

`docs/reference/general-player.md` §4.2 *Q502d*, §3.3. What is held down here:

- **The check plays end to end** on a five-square game at a tiny budget, through the core,
  and is a pure function of its inputs: the same game, roster and rule give the same file.
- **It carries its own basis**, and a mismatch is stale with a reason, never a verdict.
- **It holds nothing out of the band** (Q645, `docs/archive/plan-confidence.md` §2.4): a failing
  verdict, current or stale, is said by the trust line (Q647) and leaves the game `human` in the
  band `calibrate` and `anchor` read, so `calibrate`'s output does not move.
- **It moves no score.** Its sources are in no fingerprint set (the frozen metrics and
  analysis fingerprints in `tests/test_measurement.py` are the rest of that proof).
- **Fairy-Stockfish.** Its half is composed into the verdict (tested with a stand-in duel),
  a reader with no binary does not stale a verdict one played, and the real duel runs only
  where a binary is installed -- never here, and never downloaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_corpus import SPEC
from test_rated_nets import ROLLED_OUT, ROUND_ROLLED_OUT
from test_rated_nets import _store as rated_store

from chess_variant_sandbox import cli, corpus, fairy, measurement, premise
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament import duel as duel_module
from chess_variant_sandbox.tournament.duel import Curtailed, Duel

ROSTER = ("random", "greedy", "minimax:nodes=12", "minimax:nodes=24")
TINY = premise.Rule(pairs=2, fairy_pairs=2)
DIGEST = spec_hash(SPEC)
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def played() -> dict[str, Any]:
    """One real check of the five-square game, shared by the tests that only read it."""
    return premise.check(SPEC, agents=ROSTER, rule=TINY, workers=1)


def _stored(tmp_path: Path, payload: dict[str, Any]) -> Path:
    runs = tmp_path / "runs"
    premise.write(payload, runs)
    return runs


def _synthetic(verdict: str, rule: premise.Rule = premise.RULE) -> dict[str, Any]:
    """A stored check whose basis is exactly today's for the canonical roster.

    For the readers only: playing the canonical top rung against eight times itself is
    Q502d2's campaign, not a unit test.
    """
    basis = premise.PremiseBasis.current(DIGEST, SPEC, rule=rule)
    score = 0.35 if verdict == premise.PASSES else 0.05
    return {
        "hash": DIGEST,
        "name": SPEC["name"],
        "verdict": verdict,
        "ceiling": {
            "candidate": basis.player,
            "reference": premise.ceiling_of(basis.player, rule.multiple),
            "score": score, "error": 0.02, "games": 32, "identical": 0,
            "pairs": [score] * 16, "floor": rule.ceiling_floor,
            "passes": verdict == premise.PASSES,
        },
        "fairy": {"played": False, "why": "stand-in"},
        "basis": basis.as_dict(),
    }


# ------------------------------------------------------------------ the players


class TestThePlayers:
    def test_the_top_rung_is_the_most_nodes_and_the_ceiling_multiplies_them(self) -> None:
        assert premise.top_rung(ROSTER) == "minimax:nodes=24"
        netted = "minimax:net=000000000001,nodes=6400"
        assert premise.top_rung(("random", "minimax:net=000000000001,nodes=100", netted)) == netted
        assert premise.ceiling_of(netted, 8) == "minimax:net=000000000001,nodes=51200"
        assert premise.fairy_of(netted) == "fairystockfish:nodes=6400"

    def test_a_roster_with_no_search_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no searching player"):
            premise.top_rung(("random", "greedy"))


class TestTheGamesOwnRoster:
    """Q912: the check's top rung is the game's measuring roster's, `corpus.canonical_agents`.

    Before Q912 the default roster was `CANONICAL_AGENTS` for every game, so a basis taken
    with that roster spelled out is what the check stamped then (same code, same reading).
    """

    def test_a_game_without_a_rated_net_is_unchanged_to_the_byte(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(corpus, "NETS_DIR", tmp_path)
        now = premise.PremiseBasis.current(DIGEST, SPEC)
        then = premise.PremiseBasis.current(DIGEST, SPEC, corpus.CANONICAL_AGENTS)
        assert json.dumps(now.as_dict()) == json.dumps(then.as_dict())
        assert now.player == corpus.CANONICAL_AGENTS[-1]

    def test_a_rated_net_is_checked_by_its_own_top_rung(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        then = premise.PremiseBasis.current(DIGEST, SPEC, corpus.CANONICAL_AGENTS)
        blob = rated_store(tmp_path, "000000000001")
        monkeypatch.setattr(corpus, "NETS_DIR", tmp_path)
        now = premise.PremiseBasis.current(DIGEST, SPEC)
        assert now.player == "minimax:net=000000000001,nodes=6400"
        assert now.net == "000000000001=net:" + hashlib.sha256(blob).hexdigest()[:12]
        assert premise.ceiling_of(now.player, 8) == "minimax:net=000000000001,nodes=51200"
        assert premise.fairy_of(now.player) == "fairystockfish:nodes=6400"
        reasons = then.stale_against(now)
        assert any("strongest player was minimax:nodes=6400" in reason for reason in reasons)
        assert any("its nets were none" in reason for reason in reasons)

    def test_on_the_library_the_stamp_moves_for_the_rated_net_games_alone(self) -> None:
        """Every committed and archived game: the basis moves exactly where a net is rated
        (`test_rated_nets.ROLLED_OUT`, and the round games of `ROUND_ROLLED_OUT`, RS), and
        only in who played.

        Taken without the spec (no tables, no Fairy-Stockfish half), which reads none of the
        roster: what is left to differ is what the roster decides (0.1 s, against 17 s with it)."""
        moved: dict[str, set[str]] = {}
        seen: set[str] = set()
        for directory in ("rulesets", "archive"):
            for path in sorted((ROOT / directory).rglob("*.json")):
                try:
                    spec = load_spec(path)
                except Exception:  # noqa: BLE001 - not a spec (an archive's index)
                    continue
                digest = spec_hash(spec)
                if digest in seen:
                    continue
                seen.add(digest)
                now = premise.PremiseBasis.current(digest, None).as_dict()
                then = premise.PremiseBasis.current(
                    digest, None, corpus.CANONICAL_AGENTS
                ).as_dict()
                differs = {key for key in now if now[key] != then[key]}
                if differs:
                    moved[digest] = differs
        assert set(moved) == set(ROLLED_OUT) | set(ROUND_ROLLED_OUT)
        for digest, differs in moved.items():
            assert differs == {"player", "net"}, digest


# ------------------------------------------------------------------ end to end


class TestTheCheck:
    def test_it_plays_both_seats_of_every_pair_and_grades_against_the_floor(
        self, played: dict[str, Any]
    ) -> None:
        ceiling = played["ceiling"]
        assert played["hash"] == DIGEST
        assert ceiling["candidate"] == "minimax:nodes=24"
        assert ceiling["reference"] == "minimax:nodes=192"
        k = ceiling["decided_after"]
        assert ceiling["planned_pairs"] == TINY.pairs and 1 <= k <= TINY.pairs
        assert ceiling["games"] == 2 * k and len(ceiling["pairs"]) == k
        assert ceiling["passes"] == (ceiling["score"] >= TINY.ceiling_floor)
        fairy_part = played["fairy"]
        assert played["verdict"] == (
            premise.PASSES
            if ceiling["passes"] and (not fairy_part["played"] or fairy_part["passes"])
            else premise.FAILS
        )
        assert played["basis"]["rule"] == TINY.spelled()
        assert TINY.spelled().endswith("; curtailed")
        assert played["basis"]["player"] == "minimax:nodes=24"

    # Slow (Q749): two six-pair checks, Fairy-Stockfish's duel in each, 11-12 s in the gate.
    @pytest.mark.slow
    @pytest.mark.parametrize("workers", [1, 4])
    def test_curtailed_it_gives_the_full_check_s_verdict(
        self, played: dict[str, Any], workers: int
    ) -> None:
        """reference/compute-economy.md §4.2: the same verdict as the sixteen pairs, the prefix's score."""
        rule = premise.Rule(pairs=6, fairy_pairs=6)
        whole = premise.check(
            SPEC, agents=ROSTER, rule=replace(rule, curtailed=False), workers=workers
        )
        cut = premise.check(SPEC, agents=ROSTER, rule=rule, workers=workers)
        assert cut["verdict"] == whole["verdict"]
        assert cut["ceiling"]["passes"] == whole["ceiling"]["passes"]
        k = cut["ceiling"]["decided_after"]
        assert k < rule.pairs, "the fixture never stops early"
        assert cut["ceiling"]["pairs"] == whole["ceiling"]["pairs"][:k]
        assert cut["basis"]["rule"] != whole["basis"]["rule"]

    # Slow (Q749): a second whole check, 2 s in the gate, most of it Fairy-Stockfish's.
    @pytest.mark.slow
    def test_it_is_a_pure_function_of_the_game_the_roster_and_the_rule(
        self, played: dict[str, Any]
    ) -> None:
        again = premise.check(SPEC, agents=ROSTER, rule=TINY, workers=2)
        assert again == played

    def test_written_it_reads_back_current_with_its_verdict(
        self, tmp_path: Path, played: dict[str, Any]
    ) -> None:
        runs = _stored(tmp_path, played)
        assert (runs / DIGEST / premise.PREMISE_FILE).exists()
        # Beside the score's own file, never in it: nothing else is written.
        assert sorted(p.name for p in (runs / DIGEST).iterdir()) == [premise.PREMISE_FILE]
        found = premise.read(DIGEST, runs, SPEC, agents=ROSTER, rule=TINY)
        assert found is not None and found.current
        assert found.verdict == played["verdict"]
        assert found.marked == (played["verdict"] == premise.FAILS)
        said = found.as_dict()
        assert said["state"] == played["verdict"] and said["reasons"] == []

    def test_no_file_is_no_verdict(self, tmp_path: Path) -> None:
        assert premise.read(DIGEST, tmp_path / "runs", SPEC) is None


# ------------------------------------------------------------------ its own basis


class TestItsBasis:
    def test_another_rule_or_roster_reads_stale_and_gives_no_verdict(
        self, tmp_path: Path, played: dict[str, Any]
    ) -> None:
        runs = _stored(tmp_path, played)
        other_rule = premise.read(DIGEST, runs, SPEC, agents=ROSTER, rule=premise.Rule())
        assert other_rule is not None and not other_rule.current
        assert other_rule.verdict is None and other_rule.state == "stale"
        assert any("another rule" in reason for reason in other_rule.reasons)

        canonical = premise.read(DIGEST, runs, SPEC)
        assert canonical is not None and not canonical.current
        assert any("strongest player was minimax:nodes=24" in r for r in canonical.reasons)

    @pytest.mark.parametrize(
        ("field", "said"),
        [
            ("engine", "the core, the agents or the game loop changed"),
            ("tables", "compiles to different tables"),
            ("net", "its nets were"),
            ("code", "the check itself changed"),
            ("fairy_code", "translation of this game changed"),
        ],
    )
    def test_each_field_that_moves_names_itself(
        self, tmp_path: Path, played: dict[str, Any], field: str, said: str
    ) -> None:
        moved = json.loads(json.dumps(played))
        moved["basis"][field] = moved["basis"][field] + "x" if moved["basis"][field] else "x"
        runs = _stored(tmp_path, moved)
        found = premise.read(DIGEST, runs, SPEC, agents=ROSTER, rule=TINY)
        assert found is not None and not found.current
        assert any(said in reason for reason in found.reasons), found.reasons

    def test_an_unreadable_stamp_is_stale(self, tmp_path: Path, played: dict[str, Any]) -> None:
        broken = dict(played, basis={"player": "x"})
        runs = _stored(tmp_path, broken)
        found = premise.read(DIGEST, runs, SPEC, agents=ROSTER, rule=TINY)
        assert found is not None and not found.current and found.verdict is None

    def test_no_binary_here_does_not_stale_a_verdict_one_played(self) -> None:
        stored = premise.PremiseBasis(
            DIGEST, "p", "e", "t", "", "r", "c", "fc", "Fairy-Stockfish 14@abc"
        )
        here = premise.PremiseBasis(DIGEST, "p", "e", "t", "", "r", "c", "fc",
                                    premise.FAIRY_ABSENT)
        assert stored.stale_against(here) == ()
        # The other way round it is stale: a binary here can play what was not played.
        assert "one is installed now" in " ".join(here.stale_against(stored))
        other = premise.PremiseBasis(DIGEST, "p", "e", "t", "", "r", "c", "fc",
                                     "Fairy-Stockfish 15@def")
        assert other.stale_against(stored)


# ------------------------------------------------------------------ what it marks


class TestWhatItMarks:
    def test_a_current_fail_is_marked_and_says_so_in_plain_words(
        self, tmp_path: Path
    ) -> None:
        runs = _stored(tmp_path, _synthetic(premise.FAILS))
        found = premise.read(DIGEST, runs, SPEC)
        assert found is not None and found.current and found.marked
        words = found.words()
        assert "not yet strong enough" in words and "0.05" in words and "0.20" in words
        # T1: nothing is held out of the scale on this check since 2026-10-01.
        assert "set the scale" not in words
        # The visitor's version (`docs/reference/plain-words.md` §3.2): a level word and why,
        # no floor and no number.
        said = found.site_words()
        assert said.startswith("How far to trust it: questioned. A copy of the strongest "
                               "computer player, given eight times longer to think, beats "
                               "it easily")
        assert "Read the score as a first guess" in said
        assert not re.search(r"\d", said) and "scale" not in said and "check" not in said
        assert found.as_dict()["site"] == said

    def test_a_stale_fail_stays_marked_until_a_check_passes(self, tmp_path: Path) -> None:
        stale = _synthetic(premise.FAILS)
        stale["basis"]["code"] = "0" * 16
        runs = _stored(tmp_path, stale)
        found = premise.read(DIGEST, runs, SPEC)
        assert found is not None and not found.current
        assert found.verdict is None and found.marked
        assert "out of date" in found.words() and "until a check passes" in found.words()
        assert "set the scale" not in found.words()
        assert found.site_words() == ("How far to trust it: not known. The last check was "
                                      "made on older computer players.")

    def test_a_pass_current_or_stale_marks_nothing(self, tmp_path: Path) -> None:
        passing = _synthetic(premise.PASSES)
        runs = _stored(tmp_path, passing)
        found = premise.read(DIGEST, runs, SPEC)
        assert found is not None and found.current and not found.marked
        assert found.words().startswith("Checked:")
        assert found.site_words().startswith("How far to trust it: early estimate.")
        passing["basis"]["code"] = "0" * 16
        premise.write(passing, runs)
        stale = premise.read(DIGEST, runs, SPEC)
        assert stale is not None and not stale.marked and stale.verdict is None
        assert not hasattr(premise, "band_hold")


class TestTheBand:
    """`cli._load_scored` is where `calibrate` and `anchor` read the band, and since Q645
    (`docs/archive/plan-confidence.md` §2.4) no premise check holds a game out of it."""

    @staticmethod
    def _corpus(root: Path) -> Path:
        """A real game with the check's spec, two more real games, a control, two mutants."""
        rows = [("ref-lancers", DIGEST, 0.90, None), ("ref-b", "b" * 16, 0.60, None),
                ("ref-c", "c" * 16, 0.70, None), ("dull", "d" * 16, 0.10, None),
                ("mut-x", "e" * 16, 0.20, {"expectation": "worse", "base": "ref-b"}),
                ("mut-y", "f" * 16, 0.61, {"expectation": "same", "base": "ref-b"})]
        for name, digest, total, meta in rows:
            run = root / name
            run.mkdir(parents=True)
            (run / "index-v11.json").write_text(json.dumps({
                "ruleset_name": name, "ruleset_hash": digest, "total": total,
                "viable": True, "weights_version": "v11",
                "raw": {"ladder_monotonicity": 1.0},
            }))
            if digest == DIGEST:
                (run / "spec.json").write_text(json.dumps(SPEC))
            if meta:
                (run / "mutant.json").write_text(json.dumps(meta))
        return root

    def test_a_failing_game_is_in_the_band(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = _stored(tmp_path, _synthetic(premise.FAILS))
        monkeypatch.setattr(cli, "RUNS_DIR", library)
        run = self._corpus(tmp_path / "corpus") / "ref-lancers"
        scored = cli._load_scored(run, "v11")
        assert scored is not None and scored[1:] == ("human", "")
        assert not hasattr(cli, "PREMISE_LABEL")

    def test_calibrate_does_not_move_when_a_check_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The library holds no `premise.json` today; one that fails changes nothing."""
        root = self._corpus(tmp_path / "corpus")

        def calibrated(library: Path) -> dict[str, Any]:
            monkeypatch.setattr(cli, "RUNS_DIR", library)
            assert cli.main(["calibrate", str(root), "--weights", "v11", "--json"]) == 0
            return json.loads(capsys.readouterr().out)

        empty = calibrated(tmp_path / "empty")
        failing = calibrated(_stored(tmp_path, _synthetic(premise.FAILS)))
        assert failing == empty
        # The failing game is the band's ceiling.
        assert empty["human_ceiling"] == 0.90 and empty["human_floor"] == 0.60

    def test_the_library_option_is_gone(self) -> None:
        for command in ("calibrate", "anchor"):
            with pytest.raises(SystemExit):
                cli.main([command, ".", "--library", "runs"])


# ------------------------------------------------------------------ no score moves


def test_the_check_is_in_no_fingerprint_set() -> None:
    """A change to the check, its grading or its thresholds stales verdicts, never a score."""
    root = measurement.ROOT
    ours = {
        (root / name).resolve()
        for name in (*premise.PREMISE_SOURCES, "chess_variant_sandbox/cli.py",
                     "chess_variant_sandbox/web/library.py",
                     "chess_variant_sandbox/site/rounds.py")
    }
    for kind in ("engine", "metrics", "analysis"):
        assert not ours & {path.resolve() for path in measurement.fingerprinted(kind)}, kind


# ------------------------------------------------------------------ Fairy-Stockfish


class TestFairyStockfish:
    def test_its_half_is_part_of_the_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a stand-in engine and duel: a pass above and a fail below make a fail."""
        monkeypatch.setattr(premise, "fairy_engine", lambda: "Stand-in@000000000000")
        monkeypatch.setattr(premise, "_fairy_plays", lambda spec: "")
        real = duel_module.curtailed_duel

        def duel(spec: dict[str, Any], candidate: str, reference: str, pairs: int,
                 verdict: Any, **options: Any) -> Curtailed:
            if reference.startswith("fairystockfish:"):
                found = Duel(candidate, reference, 0.0, 0.0, (0.0,) * 2, 0)
                return Curtailed(found, verdict(found.pairs + (0.0,) * (pairs - 2)), pairs)
            return real(spec, candidate, reference, pairs, verdict, **options)

        monkeypatch.setattr(duel_module, "curtailed_duel", duel)
        lenient = premise.Rule(pairs=2, fairy_pairs=2, ceiling_floor=0.0)
        payload = premise.check(SPEC, agents=ROSTER, rule=lenient, workers=1)
        assert payload["ceiling"]["passes"] is True
        assert payload["fairy"]["played"] is True
        assert payload["fairy"]["reference"] == "fairystockfish:nodes=24"
        assert payload["fairy"]["passes"] is False
        assert payload["verdict"] == premise.FAILS
        assert payload["basis"]["fairy_engine"] == "Stand-in@000000000000"

    def test_a_reach_goal_is_asked_like_any_other_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q915 lifted Q912's guard: a reach game plays Fairy-Stockfish wherever
        `fairy.checked` passes it, and the check is what refuses a goal the build cannot
        read (`fairy/unread.py`)."""
        reach = {"piece": "K", "squares": ["c5"], "mirror": False, "result": "win"}
        spec = {**SPEC, "rules": {**SPEC["rules"], "reach": reach}}
        assert not hasattr(premise, "REACH_BLIND")
        monkeypatch.setattr(fairy, "checked", lambda spec: object())
        assert premise._fairy_plays(spec) == ""

    def test_a_game_it_does_not_translate_leaves_it_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(premise, "_fairy_translates", lambda spec: False)
        basis = premise.PremiseBasis.current(DIGEST, SPEC, ROSTER, TINY)
        assert (basis.fairy_code, basis.fairy_engine) == ("", "")

    @pytest.mark.skipif(
        not fairy.available(),
        reason="no Fairy-Stockfish binary here; Q502d2 runs this locally "
        "(general-player.md §4.2, Q502d)",
    )
    @pytest.mark.slow  # a whole check against the engine, 2.2 s in the gate (Q749)
    def test_the_real_duel_where_a_binary_is_installed(self) -> None:
        if not fairy.playable(SPEC):
            pytest.skip("the installed Fairy-Stockfish does not play this game as ours does")
        payload = premise.check(SPEC, agents=ROSTER, rule=TINY, workers=1)
        part = payload["fairy"]
        assert part["played"] is True and part["games"] == 2 * part["decided_after"]
        assert part["planned_pairs"] == TINY.fairy_pairs
        assert part["reference"] == "fairystockfish:nodes=24"
        assert payload["basis"]["fairy_engine"].startswith(fairy.engine_name())


# ------------------------------------------------------------------ the pages


class TestThePages:
    """The check is no longer drawn on its own: the trust line says it among the rest of
    the evidence (Q647, `tests/test_trust_line.py`), and its maintainer's words ride in
    that line's numbers."""

    def test_the_check_travels_in_the_trust_line_only(self, tmp_path: Path) -> None:
        from test_web import TestTheIndexBand

        from chess_variant_sandbox.web.server import App

        rulesets, runs = tmp_path / "rulesets", tmp_path / "runs"
        rulesets.mkdir()
        runs.mkdir()
        (rulesets / "lancers.json").write_text(json.dumps(SPEC), encoding="utf-8")
        TestTheIndexBand._score(SPEC, DIGEST, runs, 0.61, 0.58, 0.64)
        premise.write(_synthetic(premise.FAILS), runs)
        app = App(rulesets, runs, archive_dir=tmp_path / "archive")
        [card] = app.get("/api/library")["games"]
        assert card["index"] == 0.61 and "premise" not in card
        line = app.get(f"/api/game/{DIGEST}")["trust"]
        assert any("not yet strong enough" in number for number in line["numbers"])


# ------------------------------------------------------------------ /admin


class TestAdminsWords:
    def test_nothing_on_file_is_not_checked_yet(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.site import rounds

        assert rounds.premise_said(DIGEST, SPEC, tmp_path) == rounds.PREMISE
        assert rounds.premise_said(DIGEST, SPEC, None) == rounds.PREMISE
        assert rounds.PREMISE.startswith("not checked yet")

    def test_a_verdict_on_file_is_read_and_a_stale_one_says_so(self, tmp_path: Path) -> None:
        from chess_variant_sandbox.site import rounds

        runs = _stored(tmp_path, _synthetic(premise.FAILS))
        assert rounds.premise_said(DIGEST, SPEC, runs).startswith("fails: Scored by players")
        stale = _synthetic(premise.PASSES)
        stale["basis"]["rule"] = "an older rule"
        premise.write(stale, runs)
        said = rounds.premise_said(DIGEST, SPEC, runs)
        assert said.startswith("stale: The strength check is out of date")
