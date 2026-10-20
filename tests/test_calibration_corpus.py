"""The labelled corpus as a command, and the base every null is read against (Q71).

`docs/archive/plan-variety.md` §5.4, box A4. The proof it names: a corpus built from two small
committed specs and one mutant, loaded by both readers -- `cli._load_scored`, which
`calibrate` and `anchor` go through, and `test_index_quality._corpus`, which the index
audit goes through -- with `mutant.json` naming its base, and `calibrate` refusing a
corpus that has no `same` entry.

The corpus is built once, on a roster small enough for the fast suite. The command's own
roster is `corpus.CANONICAL_AGENTS`, which plays 256 games an entry; `build` reads it at
call time so the test can swap it, and nothing else differs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import test_index_quality

from chess_variant_sandbox import calibration_corpus, cli
from chess_variant_sandbox.corpus import (
    LIBRARY_SEEDS,
    VERDICT_RULE,
    CorpusError,
    measurement_dir,
    run_dir,
)
from chess_variant_sandbox.measurement import Basis, fingerprinted, tables_fingerprint
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS
from chess_variant_sandbox.spec import load_spec, spec_hash
from chess_variant_sandbox.tournament import TournamentConfig, run_tournament

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"

# Four rungs, so the ladder has something to order and the small game passes its gates;
# two games a pairing, so the whole build is a few seconds.
ROSTER = ("random", "greedy", "minimax:nodes=50", "minimax:nodes=200")
GAMES = 2
# Two small committed specs: the first has a current library run to adopt, the second
# has none and is played. The mutant is a null of the first, so it has a base to name.
ADOPTED, PLAYED = "gardner", "dobutsu"
MUTANT = "null-renamed-gardner"
VERSION = DEFAULT_WEIGHTS


def _stamp_library(spec: dict[str, Any], library: Path, **moved: str) -> Path:
    """A library measurement of ``spec`` on ``ROSTER``, stamped with the basis of now.

    What `chess-variant-sandbox measure` would leave, without the four tournaments: the seed-0 run's
    games, and `runs/<hash>/canonical.json` with a basis. ``moved`` overrides fields of
    that basis, to stand in for a measurement taken under other code.
    """
    digest = spec_hash(spec)
    source = run_dir(digest, 0, library)
    if not (source / "games.jsonl").exists():
        run_tournament(
            spec, TournamentConfig(agents=ROSTER, games_per_pairing=GAMES, seed=0, workers=1)
        ).save(source)
    basis = Basis.current(
        ruleset_hash=digest,
        weights_version=DEFAULT_WEIGHTS,
        agents=ROSTER,
        games_per_pairing=GAMES,
        seeds=LIBRARY_SEEDS,
        tables=tables_fingerprint(spec),
        verdict_rule=VERDICT_RULE,
    ).as_dict()
    basis.update(moved)
    target = measurement_dir(digest, library)
    target.mkdir(parents=True, exist_ok=True)
    (target / "canonical.json").write_text(
        json.dumps({
            "hash": digest,
            "runs": [{"seed": seed} for seed in LIBRARY_SEEDS],
            "basis": basis,
        }),
        encoding="utf-8",
    )
    return source


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The corpus, built once through the command, and everything it was built from."""
    root = tmp_path_factory.mktemp("labelled")
    rulesets = root / "rulesets"
    rulesets.mkdir()
    for stem in (ADOPTED, PLAYED):
        shutil.copyfile(RULESETS / f"{stem}.json", rulesets / f"{stem}.json")
    library = root / "library"
    source = _stamp_library(load_spec(rulesets / f"{ADOPTED}.json"), library)
    untouched = _snapshot(library)

    runs = root / "runs"
    out = runs / "v-test"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(calibration_corpus, "CANONICAL_AGENTS", ROSTER)
        patch.setattr(calibration_corpus, "CANONICAL_GAMES", GAMES)
        code = cli.main([
            "calibration-corpus", str(out),
            "--rulesets", str(rulesets), "--library", str(library),
            "--mutant", MUTANT, "--score", VERSION, "--workers", "1",
        ])
    assert code == 0
    return {
        "root": root, "rulesets": rulesets, "library": library, "source": source,
        "runs": runs, "out": out, "untouched": untouched,
    }


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


ENTRIES = (f"ref-{ADOPTED}", f"ref-{PLAYED}", f"mut-{MUTANT}")


class TestTheLayout:
    def test_every_entry_holds_what_the_readers_open(self, world) -> None:
        out = world["out"]
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == sorted(ENTRIES)
        for entry in ENTRIES:
            for name in (
                "spec.json", "games.jsonl", "tournament.json", "analysis.json",
                f"index-{VERSION}.json",
            ):
                assert (out / entry / name).exists(), f"{entry} has no {name}"
            assert (out / entry / "mutant.json").exists() == entry.startswith("mut-")

    def test_mutant_json_names_its_base(self, world) -> None:
        """Q71. The base is the base game's `name`, which is what `calibrate` joins on."""
        info = json.loads((world["out"] / f"mut-{MUTANT}" / "mutant.json").read_text(encoding="utf-8"))
        base = load_spec(world["out"] / f"ref-{ADOPTED}" / "spec.json")
        assert info["expectation"] == "same"
        assert info["base"] == base["name"] != ""
        assert info["base_hash"] == spec_hash(base)

    def test_a_current_library_run_is_adopted_byte_for_byte(self, world) -> None:
        for name in ("games.jsonl", "tournament.json"):
            assert (world["out"] / f"ref-{ADOPTED}" / name).read_bytes() == (
                world["source"] / name
            ).read_bytes()
        entries = json.loads((world["out"] / "corpus.json").read_text(encoding="utf-8"))["entries"]
        assert entries[f"ref-{ADOPTED}"]["origin"] == "adopted"
        assert entries[f"ref-{PLAYED}"]["origin"] == "played"
        assert "no measurement" in entries[f"ref-{PLAYED}"]["reason"]
        assert entries[f"mut-{MUTANT}"]["origin"] == "played"

    def test_every_entry_is_played_or_adopted_on_the_roster_asked_for(self, world) -> None:
        for entry in ENTRIES:
            played = json.loads((world["out"] / entry / "tournament.json").read_text(encoding="utf-8"))
            assert played["config"]["agents"] == list(ROSTER)
            assert played["config"]["games_per_pairing"] == GAMES
            assert played["config"]["seed"] == 0

    def test_the_library_is_only_read(self, world) -> None:
        assert _snapshot(world["library"]) == world["untouched"]

    def test_nothing_is_left_partial(self, world) -> None:
        assert not list(world["out"].glob("*.partial"))


class TestBothReadersLoadIt:
    def test_cli_load_scored_labels_every_entry(self, world) -> None:
        read = {
            entry: cli._load_scored(world["out"] / entry, VERSION) for entry in ENTRIES
        }
        assert all(value is not None for value in read.values())
        labels = {entry: value[1:] for entry, value in read.items() if value}
        assert labels == {
            f"ref-{ADOPTED}": ("human", ""),
            f"ref-{PLAYED}": ("human", ""),
            f"mut-{MUTANT}": ("same", ADOPTED),
        }

    def test_the_index_audit_finds_it_as_one_corpus(self, world, monkeypatch) -> None:
        monkeypatch.setattr(test_index_quality, "RUNS", world["runs"])
        monkeypatch.setattr(test_index_quality, "MIN_CORPUS_RUNS", len(ENTRIES))
        monkeypatch.setattr(test_index_quality, "MIN_LABELLED_RUNS", 1)
        assert test_index_quality._corpus(VERSION) == world["out"]
        labels = {
            run["ruleset_name"]: run["_label"]
            for run in test_index_quality._labelled_runs(VERSION)
        }
        assert labels == {ADOPTED: "human", PLAYED: "human", MUTANT: "same"}


class TestCalibrate:
    def test_the_null_is_read_against_its_base(self, world, capsys) -> None:
        """A rename changes no game, so the gap is nothing at all."""
        assert cli.main(["calibrate", str(world["out"]), "--weights", VERSION, "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["detail"]["nulls"] == 1
        assert report["detail"]["null_gaps"] == {MUTANT: 0.0}

    def test_every_index_in_the_table_names_its_runs(self, world, capsys) -> None:
        """Q416. Each entry is one tournament at seed 0, and its row says one run."""
        assert cli.main(["calibrate", str(world["out"]), "--weights", VERSION]) == 0
        table = capsys.readouterr().out.split("+/-  runs\n", 1)[1]
        rows = [line for line in table.splitlines() if line.strip()]
        assert len(rows) == len(ENTRIES)
        for row in rows:
            assert "  1 run" in row, row

    def test_a_corpus_with_no_same_entry_refuses_by_name(self, world, tmp_path, capsys) -> None:
        copy = tmp_path / "no-nulls"
        for entry in ENTRIES[:2]:
            shutil.copytree(world["out"] / entry, copy / entry)
        assert cli.main(["calibrate", str(copy), "--weights", VERSION]) == 2
        error = capsys.readouterr().err
        assert "null_stability" in error and "`same` entry" in error

    def test_a_null_whose_base_is_empty_refuses_and_is_named(
        self, world, tmp_path, capsys
    ) -> None:
        """The `runs/v9` shape: the label kept, the base lost."""
        copy = tmp_path / "v9-shape"
        shutil.copytree(world["out"], copy)
        meta = copy / f"mut-{MUTANT}" / "mutant.json"
        meta.write_text(json.dumps({"expectation": "same", "base": ""}), encoding="utf-8")
        assert cli.main(["calibrate", str(copy), "--weights", VERSION]) == 2
        assert f"{MUTANT} (base '')" in capsys.readouterr().err


class TestASecondCall:
    def test_the_same_call_keeps_what_the_first_wrote(self, world, monkeypatch) -> None:
        before = {
            entry: (world["out"] / entry / "games.jsonl").read_bytes() for entry in ENTRIES
        }
        monkeypatch.setattr(calibration_corpus, "run_tournament", _never)
        monkeypatch.setattr(calibration_corpus, "analyse_corpus", _never)
        written = calibration_corpus.build(
            world["out"], ruleset_dir=world["rulesets"], library=world["library"],
            only=(MUTANT,), agents=ROSTER, games_per_pairing=GAMES, workers=1,
        )
        assert [w.origin for w in written] == ["kept"] * len(ENTRIES)
        for entry, games in before.items():
            assert (world["out"] / entry / "games.jsonl").read_bytes() == games

    def test_another_roster_into_the_same_directory_is_refused(self, world) -> None:
        with pytest.raises(CorpusError, match="games_per_pairing"):
            calibration_corpus.build(
                world["out"], ruleset_dir=world["rulesets"], library=world["library"],
                only=(MUTANT,), agents=ROSTER, games_per_pairing=GAMES + 1,
            )

    def test_a_directory_it_did_not_write_is_refused(self, world, tmp_path) -> None:
        (tmp_path / "somebody-else").mkdir()
        with pytest.raises(CorpusError, match="new directory"):
            calibration_corpus.build(
                tmp_path, ruleset_dir=world["rulesets"], library=world["library"],
                only=(MUTANT,), agents=ROSTER, games_per_pairing=GAMES,
            )


def _never(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("an entry already written was written again")


class TestAdoption:
    def test_a_library_run_played_on_another_core_is_played_again(
        self, world, tmp_path
    ) -> None:
        spec = load_spec(world["rulesets"] / f"{ADOPTED}.json")
        shutil.copytree(world["library"], tmp_path / "library")
        _stamp_library(spec, tmp_path / "library", engine="another-core")
        source, reason = calibration_corpus.adoptable(
            spec, tmp_path / "library", ROSTER, GAMES
        )
        assert source is None
        assert "replay" in reason and "core" in reason

    def test_a_library_run_on_another_roster_is_played_again(self, world) -> None:
        spec = load_spec(world["rulesets"] / f"{ADOPTED}.json")
        source, reason = calibration_corpus.adoptable(
            spec, world["library"], ROSTER[:-1], GAMES
        )
        assert source is None and "roster" in reason


class TestMembership:
    def test_every_mutant_over_the_committed_games_is_a_member(self) -> None:
        members, unbuilt = calibration_corpus.membership(RULESETS)
        mutants = [m for m in members if m.mutant is not None]
        assert mutants and not unbuilt
        assert all(m.mutant is not None and m.mutant.base for m in mutants)
        assert len({m.name for m in members}) == len(members)

    def test_an_unknown_mutant_is_refused_by_name(self) -> None:
        with pytest.raises(CorpusError, match="no-such-mutant"):
            calibration_corpus.membership(RULESETS, ("no-such-mutant",))

    def test_a_mutant_asked_for_without_its_base_is_refused(self, tmp_path) -> None:
        shutil.copyfile(RULESETS / f"{PLAYED}.json", tmp_path / f"{PLAYED}.json")
        with pytest.raises(CorpusError, match=f"{MUTANT} cannot be built"):
            calibration_corpus.membership(tmp_path, (MUTANT,))

    def test_the_writer_is_in_no_fingerprint_set(self) -> None:
        """It decides no game and computes no component, so editing it stales nothing."""
        for kind in ("engine", "metrics", "analysis"):
            assert all(p.name != "calibration_corpus.py" for p in fingerprinted(kind))
