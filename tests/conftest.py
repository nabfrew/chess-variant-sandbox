from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import site_store

import chess_variant_sandbox.engine.compile as compile_module
from chess_variant_sandbox.agents.fitted import STORE_ENV
from chess_variant_sandbox.engine import CompiledSpec, Engine
from chess_variant_sandbox.spec import load_spec

REPOSITORY = Path(__file__).resolve().parents[1]
RULESETS = REPOSITORY / "rulesets"


def one_core_per_worker() -> None:
    """Under `-n`, each xdist worker is one of the cores the `suite` lease booked.

    The code under test reads "as many threads as there are cores" in four places
    (`smoke_test`, `TournamentConfig.resolved_workers`, `analyse_corpus`, the designer's
    playouts), all as `max(1, os.cpu_count() - 1)`. Q377c counted 152 such calls in one fast
    run, most through the web app's smoke gate: at `-n 4` each worker would burst to fifteen
    threads, and the booked four cores to sixty. So a worker is told it has one core.

    It changes no answer -- threads never do; the schedule fixes every seed first -- and a
    test that compares worker counts names them. A serial run is untouched.
    """
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return
    os.cpu_count = lambda: 1
    if hasattr(os, "process_cpu_count"):  # 3.13: what `concurrent.futures` sizes pools by
        os.process_cpu_count = lambda: 1


one_core_per_worker()

# `--skip-if-green`, and the stamp every green fast-suite run leaves for it. `pytester`
# is how `test_green_stamp.py` runs a suite inside a suite.
pytest_plugins = ("green_stamp", "pytester")


class CompileMemo:
    """One `compile_spec` per distinct spec per session, where the suite used to pay for many.

    Q377 measured it: 2791 of the fast suite's 3710 compiles were of a spec it had already
    compiled, 88 s of a 384 s run, because a dozen tests each walk the whole corpus and
    most modules build their own engines. `CompiledSpec` is a frozen dataclass of tuples
    and frozensets, so one compiled spec can be handed to every test that asks for it.

    Keyed by the spec's whole JSON, `name` included, since the compiled spec carries its
    name. Only a successful compile is kept: a spec that raises raises every time.

    **What it keeps is bounded**, because a compiled board holds a table per square per
    piece: the 71-spec corpus is 187 MB, a wrapped board up to 14 MB. So a spec is kept
    only the *second* time it is compiled -- the corpus walks and the shared engines come
    back, the one-off variants (54 file shifts of the wrapped boards, every mutation the
    loop tests draw) do not -- and at most `KEPT` of them, least recently used out first.
    Keeping everything took a run's peak memory from 290 MB to 938 MB.

    It stands in for the name in every module that holds it, which is `compile.py` itself
    (for the modules that import it inside a function), the `chess_variant_sandbox.engine` package,
    `engine.py` (whose `Engine.__init__` compiles a dict) and any other module already
    imported. It is installed when this file is imported, before any test module is, so a
    test module's own `from chess_variant_sandbox.engine import compile_spec` gets it too.

    A test that changes what compiling means -- monkeypatching a compile internal, then
    compiling a spec it compiled before -- would be handed the old tables. Such a test
    takes `@pytest.mark.fresh_compile`, and every compile it makes is a real one.
    """

    KEPT = 128

    def __init__(self, real: Any) -> None:
        self.real = real
        self.kept: OrderedDict[str, CompiledSpec] = OrderedDict()
        self.seen: set[str] = set()
        self.bypass = False
        self.hits = self.misses = 0

    def __call__(self, spec: dict[str, Any]) -> CompiledSpec:
        if self.bypass:
            return self.real(spec)
        try:
            key = json.dumps(spec, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return self.real(spec)
        compiled = self.kept.get(key)
        if compiled is not None:
            self.hits += 1
            self.kept.move_to_end(key)
            return compiled
        self.misses += 1
        compiled = self.real(spec)
        if key not in self.seen:
            self.seen.add(key)
            return compiled
        self.kept[key] = compiled
        if len(self.kept) > self.KEPT:
            self.kept.popitem(last=False)
        return compiled

    def install(self) -> None:
        for name, module in list(sys.modules.items()):
            if name.startswith("chess_variant_sandbox") and getattr(module, "compile_spec", None) is self.real:
                module.compile_spec = self  # type: ignore[attr-defined]


compile_memo = CompileMemo(compile_module.compile_spec)
compile_memo.install()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fresh_compile: compile every spec for real, bypassing the session's compile memo",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    compile_memo.bypass = item.get_closest_marker("fresh_compile") is not None


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    compile_memo.bypass = False


# The three stores git tracks. Each is a directory of evidence keyed by ruleset hash --
# what an opponent was given to play one game with -- so what makes them worth anything
# is that every file in them was put there on purpose.
TRACKED_STORES = ("fitted", "heuristics", "valuations")


@pytest.fixture(scope="session", autouse=True)
def fits_are_written_somewhere_temporary(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """A green run must leave the tree clean, and until Q91 it did not.

    `corpus.measure` fits a candidate on its way to measuring it, and `ensure_fit` is the
    one writer whose caller names no path -- so `test_corpus.py`'s five-square fixture
    wrote `fitted/ee26d2322244-*.json` into the store git tracks, where a `git add -A`
    could commit it, and once did. This points that default at a temporary directory for
    the session.

    **Writes only.** `load_fit(hash)` with no root still reads `fitted/`, because tests
    that ask whether chess has a committed fit are asking about the repository's store
    and would otherwise turn silently into skips.
    """
    store = tmp_path_factory.mktemp("fit-store")
    previous = os.environ.get(STORE_ENV)
    os.environ[STORE_ENV] = str(store)
    try:
        yield store
    finally:
        if previous is None:
            os.environ.pop(STORE_ENV, None)
        else:
            os.environ[STORE_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def tables_are_cached_somewhere_temporary(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """`measurement.tables_fingerprint` keeps its answers in `runs/`; a test run must not.

    A `runs/` with only the cache in it would stop a test that skips on an empty `runs/`
    from skipping. An environment variable rather than a patched constant, so a worker a
    test spawns caches in the same place.
    """
    from chess_variant_sandbox.measurement import TABLES_CACHE_ENV

    cache = tmp_path_factory.mktemp("tables-cache") / "tables-fingerprints.json"
    previous = os.environ.get(TABLES_CACHE_ENV)
    os.environ[TABLES_CACHE_ENV] = str(cache)
    try:
        yield cache
    finally:
        if previous is None:
            os.environ.pop(TABLES_CACHE_ENV, None)
        else:
            os.environ[TABLES_CACHE_ENV] = previous


@pytest.fixture(scope="session", autouse=True)
def the_tracked_stores_do_not_move() -> Iterator[None]:
    """Nothing the suite runs may add to, or remove from, a store git tracks.

    The fixture above closes the one leak that was found; this is what would find the
    next one, and it is deliberately about the *directories* rather than about
    `ensure_fit`. A store is written through several paths -- `chess-variant-sandbox fit`, `chess-variant-sandbox
    values`, the browser's opponent builder -- and any of them reached with a default
    root writes into the repository. Comparing the listing before and after turns "a
    green run leaves the tree clean" into something the suite itself asserts.

    Names, not contents: a test that rewrote a committed fit with identical bytes is not
    a defect worth failing a run over, and reading 186 files twice a session is not worth
    paying for. `git status` is the check for content; this is the check for provenance.
    """
    def listing() -> dict[str, frozenset[str]]:
        return {
            name: frozenset(
                str(path.relative_to(REPOSITORY))
                for path in (REPOSITORY / name).rglob("*")
                if path.is_file()
            )
            for name in TRACKED_STORES
        }

    before = listing()
    yield
    after = listing()
    for name in TRACKED_STORES:
        added = sorted(after[name] - before[name])
        removed = sorted(before[name] - after[name])
        assert not added, f"the suite wrote into the tracked {name}/ store: {added}"
        assert not removed, f"the suite deleted from the tracked {name}/ store: {removed}"


@pytest.fixture(autouse=True)
def registrations_end_with_their_test() -> Iterator[None]:
    """What a test registers by name is gone when it ends (Q431).

    `heuristics._HEURISTICS`, `valuations._VALUATIONS` and `fitted._FITTED` are
    module-level lookups an agent string resolves through, and nothing ever empties them.
    So a test's answer could depend on what an earlier test in the same process
    registered. Under `-n 4` that is whichever tests xdist happened to hand the same
    worker, which is how Q431 was seen and not reproduced for three nights.

    The case that was found: the heuristic digest is taken over the *unit-length* weights
    and `register_heuristic` keeps the first vector registered under one, while the core
    plays the raw weights and is not bitwise scale-invariant. `test_backends.py`'s hand
    arms on gardner register `builtin_vector` with its hand weight zeroed, which is
    gardner's own built-in vector at about 0.707 of unit length, under gardner's own
    digest. Run before `test_valuations.py`'s null arm, that null arm's `plain` resolved
    to the shorter vector, played a different game, and scored 0.4167 with 3 of 6 pairs
    identical instead of exactly 0.5. The library defect behind it is written up in the
    Q431 report; this keeps any one test from inheriting another's players. Q433 closed
    that defect: registration refuses a second raw vector under a held digest, and the
    hand arms spell their hand-free player as gardner's own built-in, to the bit.

    Restored in place, on the dict objects held at setup, so a module that imported a
    registry by name still sees the one the resolvers read, and a `monkeypatch` that
    swapped one for the length of a test has already put the original back when this
    runs. Registrations made in a module- or session-scoped fixture predate the snapshot
    and survive it, as that fixture's scope says they should.
    """
    from chess_variant_sandbox.agents import fitted, heuristics, valuations

    held = [
        (registry, dict(registry))
        for registry in (heuristics._HEURISTICS, valuations._VALUATIONS, fitted._FITTED)
    ]
    yield
    for registry, before in held:
        registry.clear()
        registry.update(before)


# The scrypt cost every test but one hashes a password at. The site's is n=2**14, 282 ms
# to hash and 323 ms to verify here (Q377), and every sign-in in the site's tests paid it.
# The stored string records its own n and `verify` reads it back, so a cheap hash is still
# a real one, checked by the same code.
CHEAP_SCRYPT_N = 1 << 4


@pytest.fixture(scope="session", autouse=True)
def scrypt_is_cheap() -> Iterator[int]:
    """Lower `identity.SCRYPT_N` for the session, and hand back the real one.

    `test_site.py`'s `TestPasswordsThemselves` keeps one round trip at the real cost,
    through `real_scrypt`, so the site's own parameters -- and OpenSSL's memory limit for
    them -- are still exercised once.
    """
    from chess_variant_sandbox.site import identity

    real = identity.SCRYPT_N
    identity.SCRYPT_N = CHEAP_SCRYPT_N
    try:
        yield real
    finally:
        identity.SCRYPT_N = real


@pytest.fixture
def real_scrypt(scrypt_is_cheap: int, monkeypatch: pytest.MonkeyPatch) -> int:
    """The site's own scrypt cost, for the one test that hashes at it."""
    from chess_variant_sandbox.site import identity

    monkeypatch.setattr(identity, "SCRYPT_N", scrypt_is_cheap)
    return scrypt_is_cheap


@pytest.fixture(scope="session", autouse=True)
def the_sessions_site_store_is_dropped() -> Iterator[None]:
    """`site_store` makes one migrated schema for the site's database tests to share,
    the first time one asks; it goes when the session does, as each test's used to."""
    yield
    site_store.drop()


@pytest.fixture(scope="session")
def warm_library() -> None:
    """List the committed corpus once in this process, before a timed request needs it.

    A process's first `/api/library` runs the smoke gate and the Fairy-Stockfish
    translation over every ruleset in `rulesets/` and `archive/` (`library._playability`,
    `library._fairy`). Both are memoised on the spec from then on, and a second listing
    takes 0.03 s. An xdist worker is told it has one core (`one_core_per_worker`), and on
    one core that first listing took 13.0 s on a quiet machine: 9.3 s of smoke gate and
    3.1 s of translation (Q415, 2026-09-24).
    `TestServer::test_one_connection_serves_several_requests` asked for it under a 10 s
    socket timeout. It failed whenever it was the first test in its worker to list the
    library, and passed wherever an earlier test had paid the cost.

    Take this fixture in a test whose request timeout bounds *serving*, and the warm-up is
    paid in setup instead. The caches are keyed by spec content, so this also warms any
    other root that holds the same specs. Test time limits count setup, so a test with a
    `timeout` mark sets `func_only=True`. A test that can skip first, for example for want
    of a database, asks for this fixture after the skip
    (`test_site_admin.py`'s `corpus_listed`).
    """
    from chess_variant_sandbox.web import library
    from chess_variant_sandbox.web.server import App

    App(library.RULESET_DIR, library.RUNS_DIR).get("/api/library")


@pytest.fixture(scope="session")
def chess_spec() -> dict:
    return load_spec(RULESETS / "chess.json")


@pytest.fixture(scope="session")
def chess(chess_spec: dict) -> Engine:
    """Compiled chess. Session-scoped: compilation builds a table per square per piece."""
    return Engine(chess_spec)
