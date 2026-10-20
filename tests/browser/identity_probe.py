"""The half of the browser identity test that runs twice: natively, and inside Pyodide.

`tests/test_browser_identity.py` builds one job and hands the same text to two hosts. The
native host is this file run by the working tree's own Python, with the working tree's
own core. The browser host is this file loaded by `identity.mjs` into Pyodide under Node,
where `chess_variant_sandbox` is the wheel from `build/browser/` and the core is its
WebAssembly build. Each answers with the text `run` returns, and the test compares the two
answers field by field.

So this file decides nothing about the games. Every record, analysis and score in a reply
comes out of the package itself, through the same calls a measurement makes --
`slices.run_request` for the canonical games and the analysis, `slices.measure_in_slices`
for a whole measurement -- and a difference between the two replies is a difference
between the two builds, never between two harnesses.

It imports nothing from the repository but the installed package, because inside Pyodide
the package under test is the wheel's and the tree is not on `sys.path`. It is stdlib and
`chess_variant_sandbox` only, and it keeps no state between tasks beyond what the package
keeps, exactly as in a worker.

    python tests/browser/identity_probe.py < job.json > reply.json
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# The files a measurement's run directory holds, and the only ones a sliced measurement
# writes. `canonical.json` is deliberately not one: nothing browser-made writes it.
MEASURE_FILES = ("games.jsonl", "tournament.json", "analysis.json")


def run(job_text: str, host: dict[str, Any] | None = None) -> str:
    """Every task of a job, in order, as one JSON reply.

    `host` is what the caller knows about itself that Python cannot see -- the Node and
    Pyodide versions, from the driver -- and is echoed back beside what Python can.
    """
    job = json.loads(job_text)
    results: dict[str, dict[str, str]] = {}
    seconds: dict[str, float] = {}
    for task in job["tasks"]:
        started = time.perf_counter()
        results[task["name"]] = TASKS[task["kind"]](task)
        seconds[task["name"]] = round(time.perf_counter() - started, 3)
    return json.dumps(
        {"host": describe(host or {}), "results": results, "seconds": seconds},
        sort_keys=True,
    )


def describe(extra: dict[str, Any]) -> dict[str, Any]:
    """What built and runs this side, so a reply says which program answered it."""
    from chess_variant_sandbox import measurement

    answer: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
        # 64 natively and 32 under Emscripten. The one difference the probe found came
        # from it, so a reply carries it rather than leaving a reader to guess.
        "hash_width": sys.hash_info.width,
        "package": str(Path(measurement.__file__).resolve().parent),
        "build": measurement.built_fingerprint(),
        "sources": {
            kind: measurement.source_fingerprint(kind)
            for kind in ("engine", "metrics", "analysis")
        },
    }
    answer.update(extra)
    return answer


# ------------------------------------------------------------------------ the tasks


def _roster() -> Any:
    """The canonical roster at one thread, which is what a browser worker has."""
    from chess_variant_sandbox.tournament import slices

    return slices.canonical_config(workers=1)


def sample(task: dict[str, Any]) -> dict[str, str]:
    """Some of the canonical schedule's games, by index, and optionally their analysis.

    The games go through `run_request` as one `tournament` unit, which is the text a
    browser worker is handed, so the games are the canonical measurement's own at those
    indices, seeds included. The analysis is the measurement's rule as well: the graded
    games among them, through one `analysis` unit built from this side's own records.
    """
    from chess_variant_sandbox.tournament import slices

    spec, config = task["spec"], _roster()
    unit = slices.Unit(
        phase=slices.TOURNAMENT, part=0, parts=1, games=tuple(task["games"])
    )
    played = slices.run_request(slices.request(unit, spec, config=config))
    answer = {"games": played}
    if task.get("analyse"):
        records = slices.parse_records(played)
        text = ""
        for piece in slices.analysis_units(records, 1):
            request = slices.request(piece, spec, records=records, config=config)
            text += slices.run_request(request)
        answer["analysis"] = text
    return answer


def games(task: dict[str, Any]) -> dict[str, str]:
    """One pairing at the seeds named, played by `play_game` -- for a player off the roster."""
    from chess_variant_sandbox.engine import Engine
    from chess_variant_sandbox.tournament.play import play_game

    engine = Engine(task["spec"])
    first, second = task["agents"]
    lines = [
        json.dumps(play_game(engine, (first, second), seed).as_dict(), separators=(",", ":"))
        for seed in task["seeds"]
    ]
    return {"games": "".join(line + "\n" for line in lines)}


def perft(task: dict[str, Any]) -> dict[str, str]:
    from chess_variant_sandbox.engine import Engine

    engine = Engine(task["spec"])
    leaves = engine.perft(engine.initial_state(), int(task["depth"]), threads=1)
    return {"leaves": str(leaves)}


def measure(task: dict[str, Any]) -> dict[str, str]:
    """A whole measurement from nothing: tournament, analysis, blunder trials, score.

    `measure_in_slices` at one slice and one thread, into a directory of its own, and the
    files it wrote read back as text. The headline goes back as `repr`, so a reader can
    tell a difference in the last bit from a difference at the fourth decimal.
    """
    from chess_variant_sandbox.tournament import slices

    with tempfile.TemporaryDirectory(prefix="identity-") as directory:
        gathered = slices.measure_in_slices(task["spec"], directory, slices=1, config=_roster())
        out = Path(directory)
        answer = {name: (out / name).read_text(encoding="utf-8") for name in MEASURE_FILES}
        index = out / f"index-{gathered.result.weights_version}.json"
        answer["index"] = index.read_text(encoding="utf-8")
    answer["headline"] = repr(gathered.index)
    return answer


TASKS = {
    "sample": sample,
    "games": games,
    "perft": perft,
    "measure": measure,
}


if __name__ == "__main__":
    # Bytes both ways, so the platform's console encoding never touches a record.
    sys.stdout.buffer.write(run(sys.stdin.buffer.read().decode("utf-8")).encode("utf-8"))
