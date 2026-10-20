"""The labelled corpus, written by a command: every committed game and every mutant.

`chess-variant-sandbox calibrate` and `chess-variant-sandbox anchor` read one directory of entries, and until this
module nothing in `chess_variant_sandbox/` wrote one. `runs/v9` was put together outside the CLI, and 30
of its 34 `mutant.json` files carry `"base": ""` -- so every null lost its base and null
stability was reported over zero nulls (Q71). This is the one writer, and it writes the
base.

**The membership** is `rulesets/` and every mutant `metrics/mutants.py` builds over it.
A committed game is labelled by its name, as `cli._load_scored` labels it; a mutant carries
`mutant.json` with its `expectation` and its `base`, the base game's `name`, which is the
key `calibration.calibrate` joins a null to its base on.

**The games.** A committed game's seed-0 run is *adopted* from the library wherever the
library's `runs/<hash>/canonical.json` is current by roster, core, build and tables: a
game is a function of `(hash, seed, agents)`, so those are the games a replay would play,
and they are copied byte for byte. Anything else -- every mutant, and a committed game the
library cannot vouch for -- is played at seed 0 on `corpus.CANONICAL_AGENTS`. The library is only ever read.

**The analysis pass runs on every entry**, adopted or played, so the whole corpus rests on
one pass. **Scoring happens only when asked**, under the version named:
`tests/test_index_quality.py` fails when two directories of `runs/` are scored under one
version, so a new corpus scored under the version `runs/v9` already carries would break
the audit that reads the old one.

Each entry is written into `<entry>.partial` and renamed when whole, and `corpus.json`
records what the corpus was written under. A second call on the same directory under the
same code finishes what the first left, and refuses anything else rather than mixing two
corpora in one directory.

Nothing here decides a game or computes a component, so this file is in no fingerprint set.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chess_variant_sandbox.corpus import (
    CANONICAL_AGENTS,
    CANONICAL_GAMES,
    CANONICAL_SEED,
    KNOWN_SEED_SETS,
    RULESET_DIR,
    RUNS_DIR,
    VERDICT_RULE,
    CorpusError,
    measurement_dir,
    run_dir,
)
from chess_variant_sandbox.measurement import (
    Basis,
    built_fingerprint,
    players_fingerprint,
    source_fingerprint,
    stale_build,
    tables_fingerprint,
)
from chess_variant_sandbox.measurement import load as load_measurement
from chess_variant_sandbox.metrics import DEFAULT_WEIGHTS, load_weights, score_ruleset
from chess_variant_sandbox.metrics.mutants import MUTANTS, Mutant, build_mutants
from chess_variant_sandbox.metrics.scoring import Weights, WeightsError
from chess_variant_sandbox.spec.canonical import load_spec, spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from chess_variant_sandbox.tournament import TournamentConfig, rate, read_records, run_tournament
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, analyse_corpus

MANIFEST = "corpus.json"
PARTIAL = ".partial"
RULESET_PREFIX = "ref-"
MUTANT_PREFIX = "mut-"
# The two files a run carries over from the library. The analysis pass is run here and the
# index file is written here, so neither is taken from a run measured under other metrics.
ADOPTED = ("games.jsonl", "tournament.json")


@dataclass(frozen=True)
class Member:
    """One entry of the corpus: a committed game, or a mutant of one."""

    directory: str
    spec: dict[str, Any]
    mutant: Mutant | None = None
    base_hash: str = ""

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    @property
    def hash(self) -> str:
        return spec_hash(self.spec)


@dataclass(frozen=True)
class Written:
    """What became of one entry, for the person watching and for `corpus.json`."""

    directory: str
    name: str
    # adopted | played | kept -- kept is an entry a previous call already wrote.
    origin: str
    # Why a committed game was played rather than adopted. Empty otherwise.
    reason: str = ""
    total: float | None = None
    # The scorer's sentence when it refused this run a number.
    unscored: str = ""


def membership(
    ruleset_dir: Path = RULESET_DIR, only: tuple[str, ...] = ()
) -> tuple[list[Member], list[str]]:
    """Every committed game and every mutant built over them, and the mutants left out.

    ``only`` narrows the mutants to the ones named; a name `metrics/mutants.py` does not
    declare, or one whose base is not in ``ruleset_dir``, is refused rather than dropped,
    because it was asked for. Without ``only``, a mutant whose base is absent is returned
    in the second list for the caller to report.
    """
    specs = [(path.stem, load_spec(path)) for path in sorted(Path(ruleset_dir).glob("*.json"))]
    if not specs:
        raise CorpusError(f"{ruleset_dir} holds no rulesets")
    bases = {str(spec["name"]): spec for _, spec in specs}

    declared = {mutant.name: mutant for mutant in MUTANTS}
    unknown = sorted(set(only) - set(declared))
    if unknown:
        raise CorpusError(
            f"metrics/mutants.py declares no mutant called {', '.join(unknown)}"
        )
    wanted = [declared[name] for name in only] if only else list(MUTANTS)
    unbuilt = sorted(mutant.name for mutant in wanted if mutant.base not in bases)
    if only and unbuilt:
        raise CorpusError(
            f"{', '.join(unbuilt)} cannot be built: {ruleset_dir} holds no game named "
            + ", ".join(sorted({declared[name].base for name in unbuilt}))
        )

    members = [Member(f"{RULESET_PREFIX}{stem}", spec) for stem, spec in specs]
    for mutant, spec in build_mutants(bases):
        if only and mutant.name not in only:
            continue
        members.append(
            Member(
                f"{MUTANT_PREFIX}{mutant.name}", spec, mutant, spec_hash(bases[mutant.base])
            )
        )
    return members, unbuilt


def _refuse_membership(members: list[Member]) -> None:
    """Every spec validates and every name is its own, before a game is played.

    `calibrate` joins a null to its base by `name`, so two entries sharing one would pair
    a null with whichever was read last.
    """
    invalid = []
    for member in members:
        errors = validate_spec(member.spec)
        if errors:
            invalid.append(f"{member.directory}: {errors[0]}")
    if invalid:
        raise CorpusError(
            "a corpus is not written around a hole, and these do not validate: "
            + "; ".join(invalid)
        )
    seen: dict[str, str] = {}
    for member in members:
        if member.name in seen:
            raise CorpusError(
                f"{seen[member.name]} and {member.directory} are both called "
                f"{member.name!r}, and calibrate reads entries by name"
            )
        seen[member.name] = member.directory


def adoptable(
    spec: dict[str, Any],
    library: Path,
    agents: tuple[str, ...],
    games_per_pairing: int,
    seed: int = CANONICAL_SEED,
) -> tuple[Path | None, str]:
    """The library's run of this game at ``seed``, or ``None`` and the reason it is played.

    Taken only where the game's `runs/<hash>/canonical.json` says a replay would play the
    same games: the same roster and games a pairing, the same core and build, the same
    tables, on a seed set the project uses. The metrics, the analysis pass and the weights
    may all have moved, since the pass is run again here and the score is taken here.
    The old single-run path, `runs/<hash>-0/canonical.json`, is never read: it is stale
    whatever it says.
    """
    digest = spec_hash(spec)
    now = Basis.current(
        ruleset_hash=digest,
        weights_version=DEFAULT_WEIGHTS,
        agents=agents,
        games_per_pairing=games_per_pairing,
        seeds=(seed,),
        tables=tables_fingerprint(spec),
        verdict_rule=VERDICT_RULE,
        known_seed_sets=KNOWN_SEED_SETS,
        players=players_fingerprint(spec, tuple(agents)),
    )
    stored = load_measurement(measurement_dir(digest, library) / "canonical.json", now)
    if stored is None:
        return None, "the library holds no measurement of it"
    if stored.basis is None:
        return None, "its measurement records no basis"
    if seed not in stored.basis.seeds:
        return None, f"its measurement has no run at seed {seed}"
    if stored.staleness.stored_needs == "replay":
        return None, stored.staleness.summary()

    source = run_dir(digest, seed, library)
    if not all((source / name).exists() for name in ADOPTED):
        return None, f"{source} does not hold its games"
    try:
        config = json.loads((source / "tournament.json").read_text(encoding="utf-8"))
        config = config.get("config") or {}
    except (OSError, ValueError, AttributeError):
        return None, f"{source}/tournament.json is not readable"
    if (
        tuple(str(agent) for agent in config.get("agents") or ()) != tuple(agents)
        or int(config.get("games_per_pairing", -1)) != games_per_pairing
        or int(config.get("seed", -1)) != seed
    ):
        return None, f"{source} holds a tournament on another roster or seed"
    return source, ""


def build(
    out: Path,
    ruleset_dir: Path = RULESET_DIR,
    library: Path = RUNS_DIR,
    only: tuple[str, ...] = (),
    score: str | None = None,
    workers: int = 0,
    report: Callable[[str], None] | None = None,
    agents: tuple[str, ...] | None = None,
    games_per_pairing: int | None = None,
) -> list[Written]:
    """Write the labelled corpus into ``out``, one directory per entry.

    ``agents`` and ``games_per_pairing`` default to the canonical roster, read at call
    time; a corpus on any other roster is not the one `calibrate` means, and they exist
    so a test can build a small one. ``score`` names a weights version to score every
    entry under, and nothing is scored without it.
    """
    say = report if callable(report) else (lambda _message: None)
    agents = tuple(CANONICAL_AGENTS if agents is None else agents)
    games = CANONICAL_GAMES if games_per_pairing is None else games_per_pairing
    seed = CANONICAL_SEED
    out = Path(out)

    members, unbuilt = membership(ruleset_dir, only)
    _refuse_membership(members)
    # A typo in the version fails now, not after an hour of games.
    weighting = load_weights(score) if score else None

    basis = {
        "agents": list(agents),
        "games_per_pairing": games,
        "seed": seed,
        "engine": source_fingerprint("engine") + "-" + built_fingerprint(),
        # The agents' source, which `engine` held until Q631: a corpus is one basis for
        # every entry, and its rosters are not read per entry as `Basis.players` reads them.
        "agents_source": source_fingerprint("agents"),
        "analysis": source_fingerprint("analysis"),
    }
    manifest = _open_manifest(out, basis, members)
    todo = [m for m in members if m.directory not in manifest["entries"]]
    if todo:
        # A core built from older sources does not fail: it plays the previous build's
        # rules, and the corpus would say it was played on this one.
        behind = stale_build()
        if behind is not None:
            raise CorpusError(behind)

    say(
        f"{len(members)} entries into {out}: "
        f"{sum(1 for m in members if m.mutant is None)} committed games and "
        f"{sum(1 for m in members if m.mutant is not None)} mutants; "
        f"{len(members) - len(todo)} already written"
    )
    if unbuilt:
        say(f"  not built, their base is not in {ruleset_dir}: {', '.join(unbuilt)}")

    written: list[Written] = []
    for member in members:
        final = out / member.directory
        if member.directory in manifest["entries"]:
            entry = Written(member.directory, member.name, "kept")
        else:
            source, reason = _write_entry(
                member, out, library, agents, games, seed, workers, say
            )
            origin = "played" if source is None else "adopted"
            manifest["entries"][member.directory] = {
                "name": member.name,
                "hash": member.hash,
                "tables": tables_fingerprint(member.spec),
                "origin": origin,
                # Where an adopted run came from, and why a committed game was played.
                "source": "" if source is None else str(source),
                "reason": reason,
            }
            # The manifest first and the rename after: an entry listed whose directory
            # is still partial is written again, and a whole directory never goes
            # unlisted.
            _save_manifest(out, manifest)
            (out / f"{member.directory}{PARTIAL}").rename(final)
            entry = Written(member.directory, member.name, origin, reason)
        if weighting is not None:
            total, unscored = _score_entry(member, final, weighting, say)
            entry = Written(
                entry.directory, entry.name, entry.origin, entry.reason, total, unscored
            )
        written.append(entry)

    adopted = sum(1 for w in written if w.origin == "adopted")
    played = sum(1 for w in written if w.origin == "played")
    say(
        f"\n{len(written)} entries in {out}: {adopted} adopted, {played} played, "
        f"{len(written) - adopted - played} kept from an earlier call"
        + (f"; scored under {weighting.version}" if weighting is not None else "")
    )
    refused = [w.directory for w in written if w.unscored]
    if refused:
        say(f"  the scorer gave no number to {len(refused)}: {', '.join(refused)}")
    return written


def _write_entry(
    member: Member,
    out: Path,
    library: Path,
    agents: tuple[str, ...],
    games: int,
    seed: int,
    workers: int,
    say: Callable[[str], None],
) -> tuple[Path | None, str]:
    """One entry, whole, into its partial directory.

    Returns the library run it was adopted from, or ``None`` and why a committed game was
    played instead (``""`` for a mutant, which is always played).
    """
    partial = out / f"{member.directory}{PARTIAL}"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    (partial / "spec.json").write_text(json.dumps(member.spec, indent=2), encoding="utf-8")

    source, reason = (
        adoptable(member.spec, library, agents, games, seed)
        if member.mutant is None
        else (None, "")
    )
    if source is not None:
        for name in ADOPTED:
            shutil.copyfile(source / name, partial / name)
        say(f"  {member.directory}: adopted from {source}")
    else:
        say(f"  {member.directory}: played" + (f" -- {reason}" if reason else ""))
        run_tournament(
            member.spec,
            TournamentConfig(
                agents=agents, games_per_pairing=games, seed=seed, workers=workers
            ),
        ).save(partial)

    records = list(read_records(partial / "games.jsonl"))
    analyse_corpus(member.spec, records, seed=CANONICAL_SEED, workers=workers).save(
        partial / "analysis.json"
    )

    if member.mutant is not None:
        (partial / "mutant.json").write_text(
            json.dumps(
                {
                    "expectation": member.mutant.expectation,
                    "base": member.mutant.base,
                    "base_hash": member.base_hash,
                    "rationale": member.mutant.rationale,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return source, reason


def _score_entry(
    member: Member, directory: Path, weighting: Weights, say: Callable[[str], None]
) -> tuple[float | None, str]:
    """Score one written entry from its files, as `chess-variant-sandbox score` would."""
    records = list(read_records(directory / "games.jsonl"))
    analysis = CorpusAnalysis.load(directory / "analysis.json")
    try:
        result = score_ruleset(
            member.spec, records, rate(records), weights=weighting, analysis=analysis
        )
    except WeightsError as error:
        # A run the scorer refuses has no index file, so the readers skip it rather than
        # reading a 0.0 that was never a score.
        say(f"  {member.directory}: no score -- {error}")
        return None, str(error)
    result.save(directory / f"index-{result.weights_version}.json")
    return float(result.total), ""


def _open_manifest(
    out: Path, basis: dict[str, Any], members: list[Member]
) -> dict[str, Any]:
    """What ``out`` already holds, or a fresh manifest; a refusal where the two differ."""
    path = out / MANIFEST
    if not path.exists():
        whole = (
            sorted(p.name for p in out.iterdir() if p.is_dir() and not p.name.endswith(PARTIAL))
            if out.is_dir()
            else []
        )
        if whole:
            raise CorpusError(
                f"{out} already holds {len(whole)} entries and no {MANIFEST}, so what they "
                "were written under is unknown; write the corpus to a new directory"
            )
        return {"basis": basis, "entries": {}}

    manifest = json.loads(path.read_text(encoding="utf-8"))
    stored = manifest.get("basis") or {}
    moved = sorted(key for key in basis if stored.get(key) != basis[key])
    if moved:
        raise CorpusError(
            f"{out} was written under another {', '.join(moved)}; one corpus is one "
            "basis, so write this one to a new directory"
        )
    wanted = {member.directory: member for member in members}
    entries: dict[str, Any] = {}
    for directory, info in (manifest.get("entries") or {}).items():
        member = wanted.get(directory)
        if member is None:
            raise CorpusError(
                f"{out / directory} is not in this corpus's membership; write this one "
                "to a new directory"
            )
        if info.get("hash") != member.hash or info.get("tables") != tables_fingerprint(
            member.spec
        ):
            raise CorpusError(
                f"{out / directory} holds another version of {member.name}; write this "
                "corpus to a new directory"
            )
        if (out / directory).is_dir():
            entries[directory] = info
    listed = set(entries)
    stray = sorted(
        p.name
        for p in out.iterdir()
        if p.is_dir() and not p.name.endswith(PARTIAL) and p.name not in listed
    )
    if stray:
        raise CorpusError(
            f"{out} holds entries {MANIFEST} does not list: {', '.join(stray)}; move them "
            "aside first"
        )
    return {"basis": basis, "entries": entries}


def _save_manifest(out: Path, manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
