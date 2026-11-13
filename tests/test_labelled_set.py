"""The labelled set's home and its honesty rules (`docs/reference/labelled-set.md` §3, R-S2b).

Two parts. `TestTheRung` pins that a game in `labelled/` is measured as a library game is --
the library's seeds, roster and waiting schedule -- while joining nothing the library counts,
with nothing played. The rest is the guard: every label names its game by today's hash, the
set stays out of the library and the site, proven-bad games stay out of `archive/`, and the
commit order that stops a game being labelled after its score is known (§3.2), as far as git
can show it. The guard reads git and fails, never skips, where the history is missing.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "labelled"
LABELS = LABELLED / "labels"
CERTIFICATES = LABELLED / "certificates"
SEAL = LABELLED / "SEAL.json"
READING = LABELLED / "reading.json"
VERIFIED = LABELLED / "verified.json"
EDGES = (0.0525, 0.1575, 0.4075)  # §2.1, fixed as numbers
FIELDS = {
    "game", "spec_hash", "label", "source", "likeness", "quarter", "likeness_commit",
    "labelled_by", "date", "measured_before_label", "seen",
}
RECORD_FIELDS = {"W", "L", "evidence", "flagged", "in_scale"}
PROOF_FIELDS = {"kind", "stratum", "side", "claim", "parent", "change", "proof", "review"}
# §2.2's table: the 21 good library members, by quarter. A change here is a change to a
# frozen section and is refused, not edited.
LIBRARY_GOOD = {
    "Q1": {"xiangqi", "shogi", "crazyhouse", "racingkings", "sittuyin", "dobutsu", "kyoto"},
    "Q2": {"courier", "grand", "placement-chess", "makruk", "omega", "shatranj", "antichess"},
    "Q3": {"horde", "kingofthehill", "duck-chess", "capablanca", "seirawan"},
    "Q4": {"atomic", "threecheck"},
}
SCORE_WORDS = re.compile(r"\b(index|indices|scored?|scores|viable|interestingness)\b", re.I)


def _quarter(likeness: float) -> str:
    return next((f"Q{i + 1}" for i, edge in enumerate(EDGES) if likeness < edge), "Q4")


def _labels() -> dict[str, dict[str, Any]]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(LABELS.glob("*.json"))
    }


def _set_specs() -> list[Path]:
    return sorted(LABELLED.glob("*.json"), key=lambda p: p.name) if LABELLED.is_dir() else []


def _spec_files() -> list[Path]:
    """The set's own spec files: every JSON at the top of `labelled/` but the two records."""
    return [p for p in _set_specs() if p.name not in (SEAL.name, READING.name, VERIFIED.name)]


def _git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        pytest.fail(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def _commits(path: Path) -> list[str]:
    """Every commit that touched this file, newest first."""
    out = _git("log", "--format=%H", "--", str(path.relative_to(ROOT)).replace("\\", "/"))
    return out.split() if out else []


def _tracked_and_clean(path: Path) -> bool:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    return not _git("status", "--porcelain", "--", rel)


# ---------------------------------------------------------------- the rung


@pytest.fixture
def ladder(tmp_path: Path) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("rulesets", "archive", "labelled", "runs")}
    for directory in dirs.values():
        directory.mkdir()
    return dirs


def _game(name: str) -> dict[str, Any]:
    """A small game written here, not one of the set's: the rung is about where, not what."""
    return {
        "name": name,
        "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}]},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["d1"]}]},
        "rules": {"win": "checkmate"},
    }


def _write(directory: Path, name: str, spec: dict[str, Any]) -> str:
    (directory / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec_hash(spec)


class TestTheRung:
    def test_a_labelled_game_is_found_on_its_own_rung(self, ladder: dict[str, Path]) -> None:
        digest = _write(ladder["labelled"], "probe", _game("probe"))
        found = corpus.find(
            digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
            archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        )
        assert found.stage == "labelled"
        assert found.path == ladder["labelled"] / "probe.json"

    def test_it_owes_the_librarys_seeds_and_waiting_schedule(
        self, ladder: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(corpus, "PLAY_WAITING", True)
        spec = _game("probe")
        digest = _write(ladder["labelled"], "probe", spec)
        found = corpus.find(
            digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
            archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        )
        assert corpus.expected_seeds(found.stage) == corpus.expected_seeds("committed")
        assert corpus.expected_seeds(found.stage) == corpus.LIBRARY_SEEDS
        assert corpus._plays_waiting(found)
        # The basis a reader holds it to is a library game's, field for field: seeds,
        # roster (`canonical_agents` reads the game, never its rung) and the schedule.
        as_labelled = corpus.current_basis(digest, spec=spec, seeds=corpus.expected_seeds("labelled"))
        as_library = corpus.current_basis(digest, spec=spec, seeds=corpus.expected_seeds("committed"))
        assert as_labelled == as_library
        assert as_labelled.waiting

    def test_measure_plays_the_librarys_seeds(
        self, ladder: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        digest = _write(ladder["labelled"], "probe", _game("probe"))
        found = corpus.find(
            digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
            archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        )
        asked: list[dict[int, str]] = []
        monkeypatch.setattr(corpus, "_measure_seeds", lambda c, r, w, n, rep, work: asked.append(work))
        corpus.measure(found, runs_dir=ladder["runs"])
        assert asked == [{seed: "replay" for seed in corpus.LIBRARY_SEEDS}]

    def test_the_library_does_not_count_it(self, ladder: dict[str, Path]) -> None:
        kept = _write(ladder["rulesets"], "kept", _game("kept"))
        probe = _game("probe")
        probe["pieces"][1]["moves"][0]["range"] = 2
        digest = _write(ladder["labelled"], "probe", probe)
        assert corpus.corpus_hashes(ladder["rulesets"], ladder["archive"]) == [kept]
        assert set(corpus.committed_candidates(ladder["rulesets"], ladder["archive"])) == {kept}
        assert digest != kept

    def test_a_library_game_reads_as_the_library_first(self, ladder: dict[str, Path]) -> None:
        digest = _write(ladder["rulesets"], "kept", _game("kept"))
        _write(ladder["labelled"], "copy", _game("copy"))
        found = corpus.find(
            digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
            archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        )
        assert found.stage == "committed"

    def test_it_is_neither_archived_nor_promoted(self, ladder: dict[str, Path]) -> None:
        digest = _write(ladder["labelled"], "probe", _game("probe"))
        found = corpus.find(
            digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
            archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        )
        with pytest.raises(corpus.CorpusError, match="labelled set"):
            corpus.archive_game(found, archive_dir=ladder["archive"], runs_dir=ladder["runs"])
        assert "labelled set" in (corpus.promotion_blocker(found, ladder["runs"], ladder["rulesets"]) or "")
        assert not list(ladder["archive"].iterdir())


# ---------------------------------------------------------------- verify.py


def _race() -> dict[str, Any]:
    """A runner one step from its goal: the first player wins on its first move."""
    return {
        "name": "race",
        "board": {"width": 3, "height": 3},
        "pieces": [{"id": "P", "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]}],
        "setup": {"mirror": False, "placements": [
            {"piece": "P", "player": 0, "squares": ["a2"]},
            {"piece": "P", "player": 1, "squares": ["c2"]},
        ]},
        "rules": {"win": "none", "stalemate": "loss",
                  "reach": {"squares": ["a3", "b3", "c3"], "piece": "P"}},
    }


class TestVerify:
    def _certificate(self, tmp_path: Path, tree: dict[str, Any]) -> Path:
        import importlib.util

        spec = _race()
        game = tmp_path / "race.json"
        game.write_text(json.dumps(spec), encoding="utf-8")
        cert = tmp_path / "race-cert.json"
        cert.write_text(json.dumps(
            {"game": str(game), "spec_hash": spec_hash(spec), "winner": 0, "tree": tree}
        ), encoding="utf-8")
        loader = importlib.util.spec_from_file_location("labelled_verify", LABELLED / "verify.py")
        assert loader is not None and loader.loader is not None
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        self.module = module
        return cert

    def test_a_certificate_that_wins_holds(self, tmp_path: Path) -> None:
        cert = self._certificate(tmp_path, {"move": "Pa2-a3"})
        assert self.module.check(cert) == 1

    def test_a_move_that_does_not_win_is_refuted(self, tmp_path: Path) -> None:
        cert = self._certificate(tmp_path, {"move": "Pa2-a3x"})
        with pytest.raises(self.module.Refuted):
            self.module.check(cert)

    def test_a_winner_moving_second_opens_with_every_first_move(self, tmp_path: Path) -> None:
        # Player 0's runner steps to a3 and wins: as a certificate for player 1 it is refuted,
        # and a tree that leaves out a legal first move is refuted too.
        cert = self._certificate(tmp_path, {"replies": {"Pa2-a3": None}})
        data = json.loads(cert.read_text(encoding="utf-8"))
        data["winner"] = 1
        cert.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(self.module.Refuted, match="ends the game"):
            self.module.check(cert)
        data["tree"] = {"replies": {}}
        cert.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(self.module.Refuted, match="not every legal move"):
            self.module.check(cert)


def test_every_certificate_and_proof_check_holds() -> None:
    """The certificates as `verify.py` last passed them, and the cheap mechanical checks."""
    import importlib.util

    loader = importlib.util.spec_from_file_location("labelled_check", LABELLED / "check.py")
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    for name, label in _labels().items():
        for check in (label.get("proof") or {}).get("checks") or []:
            if check["type"] in ("line", "recipe"):
                ok, says = module.run(check, module._engine(ROOT / label["game"]), ROOT)
                assert ok, f"{name}: {check['type']} {says}"


# ---------------------------------------------------------------- the guard


def test_the_set_exists_with_its_good_library_members() -> None:
    labels = _labels()
    assert labels, "labelled/labels/ holds no label"
    library = {
        name: label for name, label in labels.items()
        if label["label"] == "good" and label["measured_before_label"]
    }
    by_quarter: dict[str, set[str]] = {}
    for name, label in library.items():
        by_quarter.setdefault(label["quarter"], set()).add(name)
    assert by_quarter == LIBRARY_GOOD


def test_every_label_has_its_fields_and_a_sound_quarter() -> None:
    for name, label in _labels().items():
        assert set(label) >= FIELDS, f"{name}: missing {sorted(FIELDS - set(label))}"
        assert label["label"] in ("good", "bad", "null"), name
        assert label["source"] in ("record", "proof"), name
        assert label["quarter"] == _quarter(label["likeness"]), name
        extra = RECORD_FIELDS if label["source"] == "record" else PROOF_FIELDS
        assert extra <= set(label), f"{name}: missing {sorted(extra - set(label))}"
        if label["source"] == "record":
            assert label["label"] == "good", f"{name}: a record labels only good games (§1.1)"
            assert label["W"]["low"] >= 2, f"{name}: a good game's lowest W is 2 or above (§1.1)"
            assert label["evidence"], name
        if label["label"] in ("bad", "null"):
            assert label["source"] == "proof", f"{name}: a bad or null label is a proof (§1.2)"


def test_every_label_names_its_game_by_todays_hash() -> None:
    """An in-place mechanics edit of a member breaks this on purpose: it is a new game."""
    for name, label in _labels().items():
        path = ROOT / label["game"]
        assert path.exists(), f"{name}: {label['game']} does not exist"
        assert spec_hash(load_spec(path)) == label["spec_hash"], f"{name}: {label['game']} changed"
        if label["measured_before_label"]:
            assert label["game"].split("/")[0] in ("rulesets", "archive"), name
        else:
            assert label["game"] == f"labelled/{name}.json", name


def test_every_spec_in_the_set_has_a_label() -> None:
    labels = _labels()
    for path in _spec_files():
        assert path.stem in labels, f"{path.name} has no label"


def test_the_set_is_not_in_the_library_nor_the_sites_listing(tmp_path: Path) -> None:
    from chess_variant_sandbox.web import library

    ours = {spec_hash(load_spec(path)) for path in _spec_files()}
    assert ours, "labelled/ holds no game of its own"
    assert not ours & set(corpus.corpus_hashes())
    assert not ours & set(corpus.committed_candidates())
    listed = {game.hash for game in library.discover(runs_dir=tmp_path, archive_dir=library.ARCHIVE_DIR)}
    assert not ours & listed


def test_proven_bad_games_stay_out_of_the_archive() -> None:
    library = set(corpus.corpus_hashes())
    for name, label in _labels().items():
        if label["label"] in ("bad", "null"):
            assert label["game"].startswith("labelled/"), name
            assert label["spec_hash"] not in library, f"{name} is in rulesets/ or archive/"
            assert label["measured_before_label"] is False, name


def test_no_spec_in_the_set_carries_a_score() -> None:
    for path in _spec_files():
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert "history" not in (spec.get("provenance") or {}), path.name
        assert not SCORE_WORDS.search(path.read_text(encoding="utf-8")), path.name


def test_each_file_is_written_once_and_each_spec_no_later_than_its_label() -> None:
    """Rule 1 of §3.2, by `git log`. A file not yet committed has no history to read."""
    if _git("rev-parse", "--is-shallow-repository") == "true":
        pytest.fail("a shallow clone cannot show a file's history")
    watched = [*_spec_files(), *sorted(LABELS.glob("*.json")), *sorted(CERTIFICATES.glob("*.json"))]
    first: dict[Path, str] = {}
    for path in watched:
        commits = _commits(path)
        assert len(commits) <= 1, f"{path.relative_to(ROOT)} was touched by {len(commits)} commits"
        if commits:
            assert _tracked_and_clean(path), f"{path.relative_to(ROOT)} is edited after its commit"
            first[path] = commits[0]
    for path in _spec_files():
        label = LABELS / path.name
        if path in first and label in first and first[path] != first[label]:
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", first[path], first[label]], cwd=ROOT,
                check=False,
            ).returncode == 0
            assert ancestor, f"{path.name} was committed after its label"


def test_proof_labels_hold_and_their_certificates_verify() -> None:
    verified = json.loads(VERIFIED.read_text(encoding="utf-8")) if VERIFIED.exists() else {}
    for name, label in _labels().items():
        if label["source"] != "proof":
            continue
        assert label["review"]["verdict"] == "holds", f"{name}: only a blind 'holds' enters"
        if label["proof"]["form"] == "certificate":
            cert = CERTIFICATES / f"{label['proof']['certificate']}.json"
            assert cert.exists(), f"{name}: no certificate"
            digest = hashlib.sha256(cert.read_bytes()).hexdigest()
            assert verified.get(cert.stem) == digest, f"{name}: verify.py has not passed it"


def test_the_seal_matches_the_files_it_lists() -> None:
    if not SEAL.exists():
        return
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    listed = {**seal.get("labels", {}), **{a["label"]: a["sha256"] for a in seal.get("additions", [])}}
    for path in sorted(LABELS.glob("*.json")):
        assert path.stem in listed, f"{path.name} is not in the seal"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == listed[path.stem], path.name


def test_before_the_seal_no_unmeasured_member_has_a_run() -> None:
    """§3.2's `unmeasured`, while it can still be checked here: this checkout's `runs/`."""
    runs = ROOT / "runs"
    if SEAL.exists() or not runs.is_dir():
        return
    names = {p.name for p in runs.iterdir()}
    for name, label in _labels().items():
        if label["measured_before_label"]:
            continue
        digest = label["spec_hash"]
        held = [n for n in names if n == digest or n.startswith(f"{digest}-")]
        assert not held, f"{name} has a run directory before the seal: {held}"


def test_the_reading_comes_after_the_seal() -> None:
    if not READING.exists():
        return
    assert SEAL.exists(), "a reading with no seal"
    sealed = _commits(SEAL)[-1]
    read = _commits(READING)[-1]
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", sealed, read], cwd=ROOT, check=False
    ).returncode == 0, "reading.json does not descend from the seal"
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    reading = json.loads(READING.read_text(encoding="utf-8"))
    assert set(reading["members"]) == set(seal["members"])
    sealed_at = _git("show", "-s", "--format=%cs", sealed)
    for name, row in reading["members"].items():
        if not _labels()[name]["measured_before_label"]:
            assert row["measured"] >= sealed_at, f"{name} was measured before the seal"
