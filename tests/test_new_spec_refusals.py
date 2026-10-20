"""A refusal for new specs only: a lone promotion rank past the board (Q434).

The rule it refuses can never fire, and one library game carried it for months:
`tunnel-horizontal` (`195564f716ec`) said `rank: 9` on an 8-high board (Q102). Q434
retired the game, and two of the user's decisions hold at once: the game is retired
(2026-09-25), and a retired game keeps its page, match replays included (2026-09-18).

So the refusal runs where a spec is *admitted* -- `validate_spec`, which the designer, the
proposer, the operators and `validate` on a file all ask -- and not where a spec is
*loaded* to build an engine, which is `compile_spec`'s `validate_or_raise`. The site draws
a retired game's page and replays its matches from its stored rules through that load, and
the calibration corpora under `runs/v9`, `runs/v11` and `runs/stage26` hold the game as
`ref-tunnel-horizontal`, which a re-score or `anchor --refresh-noise` compiles.

`_retired_tunnel_horizontal.json` beside this file is the spec byte for byte as it stood
in `rulesets/` before Q434.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.cli import main
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.measurement import tables_fingerprint
from chess_variant_sandbox.metrics.components import spec_facts
from chess_variant_sandbox.site import kept
from chess_variant_sandbox.spec import load_spec, spec_hash, validate_spec
from chess_variant_sandbox.web import designing, library

ROOT = Path(__file__).resolve().parents[1]
RETIRED = Path(__file__).parent / "_retired_tunnel_horizontal.json"
# The game's identity and its compiled tables, as `tests/test_schema_nine.py::FROZEN` froze
# them before Q434 took the row out.
RETIRED_HASH = "195564f716ec"
RETIRED_TABLES = "d974a81ecff48da0"
REFUSAL = "rules.promotion.rank 9 is not a rank on this board"
LIBRARY = sorted(
    (path for folder in ("rulesets", "archive") for path in (ROOT / folder).glob("*.json")),
    key=lambda path: (path.parent.name, path.stem),
)


def _retired() -> dict[str, Any]:
    return load_spec(RETIRED)


def _main_checkout() -> Path:
    """The checkout whose `runs/` holds the calibration corpora.

    A worktree's `.git` is a one-line file naming `<main>/.git/worktrees/<name>`, the same
    reading `tests/test_index_quality.py` makes.
    """
    pointer = ROOT / ".git"
    if pointer.is_file():
        gitdir = Path(pointer.read_text(encoding="utf-8").partition(":")[2].strip())
        if gitdir.parent.name == "worktrees" and gitdir.parents[1].name == ".git":
            return gitdir.parents[2]
    return ROOT


def _chess_promoting_on(rank: int) -> dict[str, Any]:
    """A new spec: chess, its promotion rank stated as a number."""
    spec = load_spec(ROOT / "rulesets" / "chess.json")
    spec["name"] = "chess promoting past the board"
    spec["rules"]["promotion"]["rank"] = rank
    return spec


# ------------------------------------------------------------------ admitting a spec


def test_the_fixture_is_the_retired_game_and_only_admission_refuses_it() -> None:
    spec = _retired()
    assert spec_hash(spec) == RETIRED_HASH
    assert validate_spec(spec) == [REFUSAL]
    assert validate_spec(spec, new=False) == []


@pytest.mark.parametrize("path", LIBRARY, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_every_library_spec_is_admitted(path: Path) -> None:
    assert validate_spec(load_spec(path)) == []


def test_validate_on_a_file_refuses_a_new_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(_chess_promoting_on(8)), encoding="utf-8")
    assert main(["validate", str(draft)]) == 1
    assert "rules.promotion.rank 8 is not a rank on this board" in capsys.readouterr().out
    # The last rank of the eight is a zone, and the same file is a game.
    draft.write_text(json.dumps(_chess_promoting_on(7)), encoding="utf-8")
    assert main(["validate", str(draft)]) == 0


def test_the_designer_s_gate_refuses_a_new_one() -> None:
    answer = designing.check(_chess_promoting_on(8))
    assert answer["valid"] is False
    assert "The rule 'promotion': on row 8 isn't a row on this board." in answer["errors"]


# ------------------------------------------------------------------ loading a stored one


def test_the_engine_loads_the_stored_game_as_it_always_was() -> None:
    """The same tables and the same opening ladder the game had in the library."""
    spec = _retired()
    assert tables_fingerprint(spec) == RETIRED_TABLES
    engine = Engine(spec)
    state = engine.initial_state()
    assert [engine.perft(state, depth) for depth in (1, 2)] == [18, 324]


def test_a_retired_game_s_page_still_draws(tmp_path: Path) -> None:
    """The site's fallback for a hash the tree no longer holds (`site/kept.py`)."""
    spec = _retired()
    game = kept.game_from(RETIRED_HASH, str(spec["name"]), spec, tmp_path)
    # The committed games only (Q749): the games like it are ranked against every game
    # it is handed, and the archive's sixty more were 1.2 s of the test's 2.3. Not on the
    # map, it is ranked against the map's games by the map's relation (M5).
    others = [one for one in library.discover(ROOT / "rulesets", tmp_path) if one.source == "committed"]
    page = library.detail(game, others)
    assert page["source"] == "retired"
    assert page["ruleLines"] and page["pieces"]
    near = library.neighbours(game, others, tmp_path)
    assert near["links"] and near["onMap"] is False


def test_a_match_on_it_still_replays() -> None:
    """`site/matches.py` rebuilds a match from `engine_for(spec)` and its stored moves."""
    engine = library.engine_for(_retired())
    state = engine.initial_state()
    for _ in range(8):
        state = engine.apply(state, engine.legal_moves(state)[0])
    assert engine.legal_moves(state)


@pytest.mark.parametrize(
    "entry",
    [
        "v9/ref-tunnel-horizontal",
        "v11/ref-tunnel-horizontal",
        "stage26/base/ref-tunnel-horizontal",
        "stage26/x8/ref-tunnel-horizontal",
    ],
)
def test_a_calibration_corpus_entry_still_loads_for_a_rescore(entry: str) -> None:
    """`spec_facts` is where `compute_components` compiles the spec, for a re-score and for
    `anchor --refresh-noise` alike; the tables are the measurement's basis."""
    stored = _main_checkout() / "runs" / entry / "spec.json"
    if not stored.is_file():
        pytest.skip(f"needs {stored}, in the main checkout's runs/")
    spec = load_spec(stored)
    assert spec_hash(spec) == RETIRED_HASH
    assert spec_facts(spec).complexity.piece_types > 0
    assert tables_fingerprint(spec) == RETIRED_TABLES
