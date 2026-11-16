"""A published game of `labelled/` or `unlike/` may join the library (R-S17g).

`docs/reference/wrong-no.md` §2 item 5: `corpus._STAYS` kept every game of the labelled set
and the unlike search out of the library, which kept a whole kind of game out of it once the
set had been read. Now a game a label calls published and played (good, by the record of
play) is promoted on `promote`'s ordinary bar -- four current runs, viable, a reason -- as a
copy: its file stays where the set was read. A designed game (no label) and a proven-bad one
or a control (a proof's label) are still refused, and nothing of either rung is archived.
Nothing here plays a game: the measurement is a stored fixture on today's basis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox import corpus
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import load_spec, spec_hash

ROOT = Path(__file__).resolve().parents[1]


def _game(name: str, reach: int = 1) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "description": "A small skirmish game.",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "name": "captain", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4",
                 "mode": "both"}]},
            {"id": "L", "name": "lancer", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both",
                 "range": reach}]},
        ],
        "setup": {"mirror": True, "placements": [
            {"piece": "K", "squares": ["c1"]}, {"piece": "L", "squares": ["a1", "e1"]}]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 120,
                  "no_progress_draw": 40},
    }


@pytest.fixture
def ladder(tmp_path: Path) -> dict[str, Path]:
    dirs = {name: tmp_path / name
            for name in ("rulesets", "archive", "labelled", "unlike", "runs", "labels")}
    for directory in dirs.values():
        directory.mkdir()
    return dirs


def _place(ladder: dict[str, Path], rung: str, name: str, spec: dict[str, Any]) -> str:
    (ladder[rung] / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")
    return spec_hash(spec)


def _label(ladder: dict[str, Path], name: str, digest: str, label: str, source: str) -> None:
    (ladder["labels"] / f"{name}.json").write_text(json.dumps({
        "game": f"labelled/{name}.json", "spec_hash": digest, "label": label, "source": source,
    }), encoding="utf-8")


def _measured(ladder: dict[str, Path], digest: str, spec: dict[str, Any], viable: bool = True,
              seeds: tuple[int, ...] = corpus.LIBRARY_SEEDS) -> None:
    """A stored measurement on today's basis, as `tests/test_corpus.py`'s fixture writes it."""
    out = corpus.measurement_dir(digest, ladder["runs"])
    out.mkdir(parents=True, exist_ok=True)
    violations = [] if viable else ["decisive_rate=0.1 below minimum 0.2"]
    (out / "canonical.json").write_text(json.dumps({
        "hash": digest, "name": spec["name"], "agents": list(corpus.CANONICAL_AGENTS),
        "games_per_pairing": corpus.CANONICAL_GAMES, "seeds": list(seeds),
        "games": 256 * len(seeds), "index": 0.8, "totals_used": len(seeds), "sd": 0.02,
        "viable": viable, "violations": violations, "weights_version": DEFAULT_WEIGHTS,
        "families": {}, "standings": [],
        "runs": [{"seed": seed, "index": 0.8, "sd": 0.02, "viable": viable,
                  "violations": violations, "unreadable": [], "origin": "played"}
                 for seed in seeds],
        "basis": corpus.current_basis(digest, spec=spec, seeds=seeds).as_dict(),
    }), encoding="utf-8")


def _find(ladder: dict[str, Path], digest: str) -> corpus.Candidate:
    return corpus.find(
        digest, runs_dir=ladder["runs"], ruleset_dir=ladder["rulesets"],
        archive_dir=ladder["archive"], labelled_dir=ladder["labelled"],
        unlike_dir=ladder["unlike"],
    )


def _blocker(ladder: dict[str, Path], found: corpus.Candidate) -> str | None:
    return corpus.promotion_blocker(found, ladder["runs"], ladder["rulesets"], ladder["labels"])


@pytest.mark.parametrize("rung", ["labelled", "unlike"])
def test_a_published_game_is_promoted_as_a_copy(ladder: dict[str, Path], rung: str) -> None:
    spec = _game("probe")
    digest = _place(ladder, rung, "probe", spec)
    _label(ladder, "probe", digest, "good", "record")
    _measured(ladder, digest, spec)
    found = _find(ladder, digest)
    assert found.stage == rung
    assert _blocker(ladder, found) is None

    written = corpus.promote(found, reason="A published game.", ruleset_dir=ladder["rulesets"],
                             runs_dir=ladder["runs"], labels_dir=ladder["labels"])
    assert written == ladder["rulesets"] / "probe.json"
    # Copied, not moved: the file the set was read in is where it was, byte for byte.
    assert (ladder[rung] / "probe.json").read_text(encoding="utf-8") == json.dumps(spec)
    assert spec_hash(json.loads(written.read_text(encoding="utf-8"))) == digest
    assert corpus.corpus_hashes(ladder["rulesets"], ladder["archive"]) == [digest]
    # The library's copy reads first from now on.
    assert _find(ladder, digest).stage == "committed"


def test_the_ordinary_bar_still_holds_for_a_published_game(ladder: dict[str, Path]) -> None:
    spec = _game("probe")
    digest = _place(ladder, "labelled", "probe", spec)
    _label(ladder, "probe", digest, "good", "record")
    found = _find(ladder, digest)
    assert "no score on the corpus roster" in (_blocker(ladder, found) or "")
    _measured(ladder, digest, spec, seeds=corpus.LIBRARY_SEEDS[:1])
    assert "out of date" in (_blocker(ladder, found) or "")
    _measured(ladder, digest, spec, viable=False)
    assert "not viable" in (_blocker(ladder, found) or "")
    _measured(ladder, digest, spec)
    with pytest.raises(corpus.CorpusError, match="needs a reason"):
        corpus.promote(found, reason=" ", ruleset_dir=ladder["rulesets"],
                       runs_dir=ladder["runs"], labels_dir=ladder["labels"])
    assert not list(ladder["rulesets"].iterdir())


@pytest.mark.parametrize(("label", "source"), [("bad", "proof"), ("null", "proof"), (None, None)])
def test_a_proven_bad_game_a_control_or_an_unlabelled_one_stays(
    ladder: dict[str, Path], label: str | None, source: str | None
) -> None:
    spec = _game("probe")
    digest = _place(ladder, "labelled", "probe", spec)
    if label is not None:
        _label(ladder, "probe", digest, label, source or "")
    _measured(ladder, digest, spec)
    found = _find(ladder, digest)
    assert "labelled set" in (_blocker(ladder, found) or "")
    with pytest.raises(corpus.CorpusError, match="labelled set"):
        corpus.promote(found, reason="any", ruleset_dir=ladder["rulesets"],
                       runs_dir=ladder["runs"], labels_dir=ladder["labels"])
    assert not list(ladder["rulesets"].iterdir())


def test_a_designed_game_stays(ladder: dict[str, Path]) -> None:
    spec = _game("probe")
    digest = _place(ladder, "unlike", "probe", spec)
    _measured(ladder, digest, spec)
    found = _find(ladder, digest)
    assert "unlike-search.md" in (_blocker(ladder, found) or "")
    with pytest.raises(corpus.CorpusError, match="unlike chess"):
        corpus.promote(found, reason="any", ruleset_dir=ladder["rulesets"],
                       runs_dir=ladder["runs"], labels_dir=ladder["labels"])


def test_a_label_for_another_game_vouches_for_nothing(ladder: dict[str, Path]) -> None:
    spec = _game("probe")
    digest = _place(ladder, "labelled", "probe", spec)
    _label(ladder, "probe", spec_hash(_game("other", reach=2)), "good", "record")
    _measured(ladder, digest, spec)
    assert "labelled set" in (_blocker(ladder, _find(ladder, digest)) or "")


@pytest.mark.parametrize("rung", ["labelled", "unlike"])
def test_nothing_of_either_rung_is_archived(ladder: dict[str, Path], rung: str) -> None:
    spec = _game("probe")
    digest = _place(ladder, rung, "probe", spec)
    _label(ladder, "probe", digest, "good", "record")
    with pytest.raises(corpus.CorpusError):
        corpus.archive_game(_find(ladder, digest), archive_dir=ladder["archive"],
                            runs_dir=ladder["runs"])
    assert not list(ladder["archive"].iterdir())


def test_the_real_labels_vouch_for_the_published_games_only() -> None:
    """Every good-by-record label of the repository's set is promotable by label, and no
    proof's label or designed game of `unlike/` is."""
    labels = [json.loads(p.read_text(encoding="utf-8"))
              for p in sorted((ROOT / "labelled" / "labels").glob("*.json"))]
    assert labels
    for label in labels:
        vouched = corpus.published_label(label["spec_hash"], ROOT / "labelled" / "labels")
        assert (vouched is not None) == (label["label"] == "good" and label["source"] == "record")
    for path in sorted((ROOT / "unlike").glob("*.json")):
        digest = spec_hash(load_spec(path))
        assert corpus.published_label(digest, ROOT / "labelled" / "labels") is None, path.name
