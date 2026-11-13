"""The unlike search's rung (`docs/reference/unlike-search.md` §5, R-S10b1).

A game designed for the search for strong games unlike chess lives in `unlike/`: measured as a
library game is -- the library's seeds, roster and waiting schedule -- and joining nothing the
library counts (`corpus_hashes`, `committed_candidates`, the site's listing, `measure` with no
argument), nor archived nor promoted. The labelled set's rung (`tests/test_labelled_set.py`)
is its twin. Nothing here plays a game.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]
UNLIKE = ROOT / "unlike"
SCORE_WORDS = re.compile(r"\b(index|indices|scored?|scores|viable|interestingness)\b", re.I)


@pytest.fixture
def ladder(tmp_path: Path) -> dict[str, Path]:
    dirs = {name: tmp_path / name for name in ("rulesets", "archive", "labelled", "unlike", "runs")}
    for directory in dirs.values():
        directory.mkdir()
    return dirs


def _game(name: str, reach: int = 1) -> dict[str, Any]:
    """A small game written here: the rung is about where, not what."""
    return {
        "name": name,
        "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "range": reach}]},
        ],
        "setup": {"placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["d1"]}]},
        "rules": {"win": "checkmate"},
    }


def _write(directory: Path, name: str, spec: dict[str, Any]) -> str:
    (directory / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec_hash(spec)


def _find(digest: str, ladder: dict[str, Path]) -> corpus.Candidate:
    return corpus.find(
        digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
        archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        unlike_dir=ladder["unlike"],
    )


def test_an_unlike_game_is_found_on_its_own_rung(ladder: dict[str, Path]) -> None:
    digest = _write(ladder["unlike"], "probe", _game("probe"))
    found = _find(digest, ladder)
    assert found.stage == "unlike"
    assert found.path == ladder["unlike"] / "probe.json"


def test_it_owes_the_librarys_seeds_roster_and_waiting_schedule(
    ladder: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(corpus, "PLAY_WAITING", True)
    spec = _game("probe")
    found = _find(_write(ladder["unlike"], "probe", spec), ladder)
    assert corpus.expected_seeds(found.stage) == corpus.LIBRARY_SEEDS
    assert corpus._plays_waiting(found)
    as_unlike = corpus.current_basis(found.hash, spec=spec, seeds=corpus.expected_seeds("unlike"))
    as_library = corpus.current_basis(found.hash, spec=spec, seeds=corpus.expected_seeds("committed"))
    assert as_unlike == as_library
    assert as_unlike.waiting


def test_measure_plays_the_librarys_seeds(
    ladder: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    found = _find(_write(ladder["unlike"], "probe", _game("probe")), ladder)
    asked: list[dict[int, str]] = []
    monkeypatch.setattr(corpus, "_measure_seeds", lambda c, r, w, n, rep, work: asked.append(work))
    corpus.measure(found, runs_dir=ladder["runs"])
    assert asked == [{seed: "replay" for seed in corpus.LIBRARY_SEEDS}]


def test_the_library_does_not_count_it(ladder: dict[str, Path]) -> None:
    kept = _write(ladder["rulesets"], "kept", _game("kept"))
    digest = _write(ladder["unlike"], "probe", _game("probe", reach=2))
    assert digest != kept
    assert corpus.corpus_hashes(ladder["rulesets"], ladder["archive"]) == [kept]
    assert set(corpus.committed_candidates(ladder["rulesets"], ladder["archive"])) == {kept}


def test_the_library_and_the_labelled_set_read_first(ladder: dict[str, Path]) -> None:
    library = _write(ladder["rulesets"], "kept", _game("kept"))
    labelled = _write(ladder["labelled"], "member", _game("member", reach=2))
    _write(ladder["unlike"], "copy", _game("copy"))
    _write(ladder["unlike"], "copy2", _game("copy2", reach=2))
    assert _find(library, ladder).stage == "committed"
    assert _find(labelled, ladder).stage == "labelled"


def test_it_is_neither_archived_nor_promoted(ladder: dict[str, Path]) -> None:
    found = _find(_write(ladder["unlike"], "probe", _game("probe")), ladder)
    with pytest.raises(corpus.CorpusError, match="unlike chess"):
        corpus.archive_game(found, archive_dir=ladder["archive"], runs_dir=ladder["runs"])
    assert "unlike-search.md" in (corpus.promotion_blocker(found, ladder["runs"], ladder["rulesets"]) or "")
    assert not list(ladder["archive"].iterdir())


# ---------------------------------------------------------------- the directory itself


def _games() -> list[Path]:
    return sorted(UNLIKE.glob("*.json"))


def test_the_rung_has_its_home_and_points_at_the_record() -> None:
    assert Path("unlike") == corpus.UNLIKE_DIR
    readme = (UNLIKE / "README.md").read_text(encoding="utf-8")
    assert "docs/reference/unlike-search.md" in readme


def test_no_unlike_game_is_in_the_library_the_labelled_set_or_the_site(tmp_path: Path) -> None:
    from chess_variant_sandbox.web import library

    ours = {spec_hash(load_spec(path)) for path in _games()}
    assert not ours & set(corpus.corpus_hashes())
    assert not ours & set(corpus.committed_candidates())
    labelled = {spec_hash(load_spec(path)) for path in (ROOT / "labelled").glob("*.json")
                if path.name not in ("SEAL.json", "reading.json", "verified.json")}
    assert not ours & labelled
    listed = {game.hash for game in library.discover(runs_dir=tmp_path, archive_dir=library.ARCHIVE_DIR)}
    assert not ours & listed


def test_no_unlike_game_carries_a_score() -> None:
    for path in _games():
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert "history" not in (spec.get("provenance") or {}), path.name
        assert not SCORE_WORDS.search(path.read_text(encoding="utf-8")), path.name
