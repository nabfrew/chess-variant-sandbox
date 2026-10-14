"""The card answers' store (Q887): the payload a process lists from stored answers is the
payload it lists by compiling, and nothing is read under a key that does not cover it.

`web/card_rules.py` works out the three per-game answers a library card carries (the
cheap gate's verdict, the Fairy-Stockfish translation, the still board) and
`library.CardAnswers` keeps them on disk under `library.card_answers_key`.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import compile as compiling
from chess_variant_sandbox.engine import engine as engine_module
from chess_variant_sandbox.loop.gates import smoke_test
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.web import card_rules, library
from chess_variant_sandbox.web.server import App

ROOT = Path(__file__).resolve().parents[1]

# A spread of what a card has to say: a pass, a refusal by the gate, rules the translator
# refuses, a hand, portals, wild pieces' walks and a hexagon.
SPREAD = (
    "rulesets/chess.json",
    "rulesets/rps-hierarchy-chess.json",
    "rulesets/torus.json",
    "rulesets/crazyhouse.json",
    "archive/ferry.json",
    "archive/wolves-patrol.json",
    "archive/hex-glinski-hexagon.json",
)

MEMOS = ("_smoke_cached", "_fairy_cached", "_engine_cached", "_still_cached")


def _cold(monkeypatch: pytest.MonkeyPatch, store: Path) -> None:
    """Every in-process memo empty, and the store the one at ``store``: what a new
    process sees, with the file a previous one left (or none)."""
    for name in MEMOS:
        warm = getattr(library, name)
        monkeypatch.setattr(library, name, lru_cache(**warm.cache_parameters())(warm.__wrapped__))
    monkeypatch.setattr(library, "_CARDS", library.CardAnswers(store))


def _counting(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    compiled: list[Any] = []
    real = compiling.compile_spec

    def counted(spec: Any, *args: Any, **kwargs: Any) -> Any:
        compiled.append(spec)
        return real(spec, *args, **kwargs)

    monkeypatch.setattr(compiling, "compile_spec", counted)
    monkeypatch.setattr(engine_module, "compile_spec", counted)
    return compiled


@pytest.fixture(scope="module")
def small_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The spread as a library of its own, with an empty `runs/`."""
    root = tmp_path_factory.mktemp("small-library")
    (root / "rulesets").mkdir()
    (root / "archive").mkdir()
    (root / "runs").mkdir()
    for name in SPREAD:
        shutil.copy(ROOT / name, root / "rulesets" / Path(name).name)
    return root


def _payload(root: Path) -> str:
    app = App(root / "rulesets", root / "runs", ROOT / "heuristics", root / "archive")
    return json.dumps(app.library_payload(), sort_keys=True)


class TestTheStoredAnswers:
    def test_the_payload_read_back_is_the_payload_compiled(
        self, small_library: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first listing compiles every game and writes the store once; a fresh
        process's listing over that store compiles nothing and answers the same bytes."""
        store = tmp_path / "library-cards.json"
        _cold(monkeypatch, store)
        compiled = _counting(monkeypatch)
        first = _payload(small_library)
        assert len(compiled) >= len(SPREAD)
        held = json.loads(store.read_text(encoding="utf-8"))
        assert held["sources"] == library.card_answers_key()
        assert len(held["cards"]) == len(SPREAD)
        assert all(set(answers) == {"smoke", "fairy", "still"}
                   for answers in held["cards"].values())

        _cold(monkeypatch, store)
        compiled.clear()
        again = _payload(small_library)
        assert compiled == []
        assert again == first
        # The spread says what it is meant to: one refusal by the gate, one by the
        # translator, and a still board for every game.
        cards = json.loads(first)["games"]
        assert sum(card["playable"] is False for card in cards) >= 1
        assert sum(card["fairyTranslates"] is False for card in cards) >= 1
        assert all(card["still"] for card in cards)

    def test_an_answer_under_another_key_is_never_read(self, tmp_path: Path) -> None:
        store = tmp_path / "library-cards.json"
        blob = json.dumps({"name": "x"}, sort_keys=True)
        name = library.CardAnswers._name(blob)
        store.write_text(json.dumps(
            {"sources": "another key", "cards": {name: {"smoke": [True, []]}}}),
            encoding="utf-8")
        answers = library.CardAnswers(store)
        assert answers.get("smoke", blob) is None
        answers.put("smoke", blob, [False, ["why"]])
        answers.save()
        held = json.loads(store.read_text(encoding="utf-8"))
        assert held["sources"] == library.card_answers_key()
        assert held["cards"] == {name: {"smoke": [False, ["why"]]}}
        assert library.CardAnswers(store).get("smoke", blob) == [False, ["why"]]

    def test_a_save_keeps_what_another_process_stored(self, tmp_path: Path) -> None:
        store = tmp_path / "library-cards.json"
        one, two = library.CardAnswers(store), library.CardAnswers(store)
        first, second = json.dumps({"a": 1}), json.dumps({"b": 2})
        one.get("smoke", first)
        two.get("smoke", second)
        one.put("smoke", first, [True, []])
        one.save()
        two.put("fairy", second, [False, ["no"]])
        two.save()
        again = library.CardAnswers(store)
        assert again.get("smoke", first) == [True, []]
        assert again.get("fairy", second) == [False, ["no"]]

    def test_a_listing_with_nothing_new_writes_nothing(self, tmp_path: Path) -> None:
        store = tmp_path / "library-cards.json"
        library.CardAnswers(store).save()
        assert not store.exists()

    def test_a_clock_cut_verdict_is_answered_but_not_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate's time budget refuses rules whose games it could not finish, which is
        about the machine's load at the time; such a verdict is never stored."""
        _cold(monkeypatch, tmp_path / "library-cards.json")
        said = ("random games of these rules take too long to finish",)
        monkeypatch.setattr(card_rules, "smoke", lambda spec, compile_rules: (False, said, True))
        blob = json.dumps(load_spec(ROOT / "rulesets/chess.json"), sort_keys=True)
        assert library._smoke_cached(blob) == (False, said)
        assert library._CARDS.get("smoke", blob) is None


class TestTheKey:
    def test_it_covers_every_module_the_answers_import(self) -> None:
        """`card_sources` holds `card_rules.py`'s whole import closure, parents included,
        read off the syntax tree rather than the pattern the key uses."""
        sources = set(library.card_sources())
        found: set[Path] = set()
        queue = [ROOT / library.CARD_ENTRY]
        while queue:
            path = queue.pop()
            if path in found:
                continue
            found.add(path)
            for node in ast.walk(ast.parse(path.read_bytes())):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
                for dotted in names:
                    parts = dotted.split(".")
                    if parts[0] != "chess_variant_sandbox":
                        continue
                    for end in range(1, len(parts) + 1):
                        target = library._module_file(".".join(parts[:end]))
                        if target is not None:
                            queue.append(target)
        assert found <= sources, sorted(str(path) for path in found - sources)
        names = {path.relative_to(ROOT).as_posix() for path in sources}
        for needed in ("chess_variant_sandbox/loop/gates.py", "chess_variant_sandbox/viz/walks.py",
                       "chess_variant_sandbox/fairy/variant.py",
                       "chess_variant_sandbox/engine/compile.py",
                       "assets/pieces/manifest.json"):
            assert needed in names, needed
        # And not the designer's or the library's own, which would make every edit to
        # either a cold start (`card_rules`' docstring).
        assert "chess_variant_sandbox/web/designing.py" not in names
        assert "chess_variant_sandbox/web/library.py" not in names

    def test_a_module_is_found_by_its_exact_name(self) -> None:
        """On a case-insensitive disk `engine.Engine` (a class) must not find
        `engine/engine.py` under a second spelling, which moved the key between
        processes with the string hashing."""
        assert library._module_file("chess_variant_sandbox.engine.Engine") is None
        found = library._module_file("chess_variant_sandbox.engine.engine")
        assert found is not None and found.name == "engine.py"

    def test_its_files_are_in_the_byte_order_of_their_paths(self) -> None:
        """Q890: sorted as `Path`s the files came out case-blind on Windows and by byte on
        the box (`CREDITS.md` after `check.py` here, before `aliases.json` there), so the
        same tree had two keys and a store written here could never be read there."""
        tree = {
            library.CARD_ENTRY: b"from chess_variant_sandbox.viz import pieces\n",
            "chess_variant_sandbox/viz/pieces.py": b"",
            "assets/pieces/aliases.json": b"{}",
            "assets/pieces/CREDITS.md": b"credits",
            "assets/pieces/svg/a.svg": b"<svg/>",
        }
        listing: dict[str, set[str]] = {}
        for name in tree:
            parent, _, base = name.rpartition("/")
            listing.setdefault(parent, set()).add(base)
        files = library.card_files_in(
            tree.get, lambda d: frozenset(listing.get(d, ())),
            [name for name in tree if name.startswith("assets/")])
        names = [name for name, _ in files]
        assert names == sorted(tree)
        assert names.index("assets/pieces/CREDITS.md") < names.index("assets/pieces/aliases.json")
        # And this checkout's own key is read the same way.
        mine = [path.relative_to(ROOT).as_posix() for path in library.card_sources()]
        assert mine == sorted(mine)

    def test_what_python_leaves_beside_the_drawings_is_not_in_it(self) -> None:
        """The main checkout's `assets/pieces/__pycache__/` (from running `check.py`) put
        bytecode in its key that no tree on the box has."""
        tree = {library.CARD_ENTRY: b"", "assets/pieces/check.py": b"print()\n"}
        clean = library.card_files_in(tree.get, lambda d: frozenset(), list(tree)[1:])
        cached = {**tree, "assets/pieces/__pycache__/check.cpython-313.pyc": b"\0\1"}
        with_cache = library.card_files_in(cached.get, lambda d: frozenset(), list(cached)[1:])
        assert with_cache == clean
        assert not any("__pycache__" in path.parts for path in library.card_sources())

    def test_it_names_the_core_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = library.card_answers_key()
        library.card_answers_key.cache_clear()
        monkeypatch.setattr(library, "built_fingerprint", lambda: "another build")
        try:
            assert library.card_answers_key() != before
        finally:
            library.card_answers_key.cache_clear()


class TestThePieceMaps:
    """`warm()`'s second pass: every game's pieces alone on an empty board
    (`library.piece_maps`), after the listing and off its path (Q890)."""

    def test_each_game_is_compiled_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Asked a piece at a time a game compiled once per piece more, and the reference
        game once per game compared with it: 661 compiles over the 87-game library."""
        for name in ("_maps_alone", "_maps_cached", "_engine_cached"):
            warm = getattr(library, name)
            monkeypatch.setattr(library, name, lru_cache(**warm.cache_parameters())(warm.__wrapped__))
        compiled = _counting(monkeypatch)
        games = library.discover(ROOT / "rulesets", ROOT / "no-runs-here", ROOT / "archive")
        games = [game for game in games if game.path.name in {Path(n).name for n in SPREAD}]
        reference = library.reference_of(games)
        assert reference is not None and len(games) == len(SPREAD)
        maps = [library.piece_maps(game, reference) for game in games]
        assert all(found for found in maps)
        assert len(compiled) == len(games)

    def test_the_maps_are_move_maps_answers(self) -> None:
        from chess_variant_sandbox.web import designing

        for name in ("rulesets/chess.json", "archive/ferry.json", "rulesets/torus.json"):
            spec = load_spec(ROOT / name)
            found = json.loads(library._maps_alone(json.dumps(spec, sort_keys=True)))
            one_by_one = [designing.move_map(spec, piece)
                          for piece in range(len(found["maps"]))]
            assert found["maps"] == [{key: one[key] for key in ("from", "value", "moves", "takes")}
                                     for one in one_by_one], name


def test_the_cards_gate_answers_the_same_on_one_thread_and_on_every_core() -> None:
    """The card plays the gate on every core (`card_rules.smoke`); its 150 games are one
    seeded stream each, so the verdict and its numbers are the one-thread verdict's."""
    for name in ("rulesets/chess.json", "rulesets/rps-hierarchy-chess.json"):
        spec = load_spec(ROOT / name)
        alone = smoke_test(spec, workers=1).as_dict()
        every = smoke_test(spec, workers=os.cpu_count() or 1).as_dict()
        assert every == alone, name


@pytest.mark.slow
def test_every_library_card_reads_back_as_it_was_worked_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole library, `rulesets/` and `archive/`: the three answers read back from the
    store are the ones worked out by compiling."""
    from chess_variant_sandbox import corpus

    store = tmp_path / "library-cards.json"
    games = library.discover(ROOT / "rulesets", tmp_path / "runs", ROOT / corpus.ARCHIVE_DIR)
    _cold(monkeypatch, store)
    worked = [(library._playability(g.spec), library._fairy(g.spec), library.still(g),
               library.piece_art(g)) for g in games]
    library.save_card_answers()
    _cold(monkeypatch, store)
    read = [(library._playability(g.spec), library._fairy(g.spec), library.still(g),
             library.piece_art(g)) for g in games]
    assert read == worked
