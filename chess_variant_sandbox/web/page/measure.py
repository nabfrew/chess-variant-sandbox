"""Measuring a draft on this computer: the coordinator on the page.

`docs/archive/plan-client-compute.md` §8, stage B5. The designer offers to measure the draft in
the browser, where the artefact `build/browser/` is served (B4). The page starts up to
`hardwareConcurrency - 1` of B3's workers, and at most four, deals them B0's units phase
by phase, gathers in one of them, and shows the index with its components under the
*measured on your machine* label. Where there is no manifest it offers nothing; the local
application answers the manifest with a quiet 204 when it has no build, because the page
asks on every open and a 404 there was a console error on every one.

**On the site it submits (B10).** The gathering half's last step also writes the upload,
tier one's slim form (`slices.upload`, stamped as played in a browser), and the page sends
it as the string it arrived as to `POST /api/provisional` when a signed-in person measured
their own saved game. It then asks `GET /api/provisional/<job>` until the server has
walked every game and scored them itself, and shows that number under its state and label
(`provisionalWords`, as the game's own page does) in place of its own, and keeps asking
while the answer's `spotCheck` is queued or running: tier two (`site/spotcheck.py`)
replays a sample of the games exactly, and a spot-checked score is what opens *Publish*,
so when the check ends the designer's *Where this game stands* is asked again (Q503e). A
mismatch is said as such: the score is not kept and the account cannot publish. A refusal named
`stale-build` means the page loaded a wheel the server no longer runs: the page fetches
the manifest again, and the next measurement loads the current one. Off the site nothing
is sent.

**On the site it measures only what the site could keep (Q1004).** The site keeps a score
from a browser for a game its designer has saved and for nothing else
(`provisional.check_owner`), and the user found it measuring a game nobody had saved, for
minutes, and then refusing the score. So the page reads the draft's standing first
(`hereKeeping`): for a visitor, for rules the library holds and for rules somebody else
saved, *measure it here* is drawn off with the sentence saying how to turn it on (sign in;
change a rule to make a game of your own). For rules nobody has saved it measures, holds the
finished upload on the page instead of sending it, and offers *save it and keep the score*,
the editor's own save; the next standing that says the game is this person's sends the
upload as it came (`hereStood`), so nothing is measured again and the site walks and scores
every game as it does any other.

**Two halves, and neither answers a question in JavaScript.** `CLAUDE.md` rule 3 holds in
the browser: what plays is the package's own core under Pyodide, and what cuts and joins
the work is the package's own `tournament/slices.py`.

- `web/static/page/measure.js` is the JavaScript half, one of the page's scripts. It starts
  workers, hands each one unit's request text at a time, puts each reply at its unit's
  place, and draws. It never reads a request, a reply, a record or a seed.
- `slices.gather_request` is the Python half, which runs in the one worker that gathers.
  Each call is one step: it takes the replies to the units it dealt last, gathers them
  with B0's own functions (`gather_analysis`, `gather_measurement`) and answers with the
  next phase's unit requests, or with the result. What it must remember between steps
  -- the settled spec and the tournament's records -- goes out in its answer as a `state`
  string, which the page hands back unread. So a step is a function of its text and
  nothing else, which is what B6 needs to resume from stored units.

**No worker is sent Python (Q458).** Both halves' Python is in the wheel: a worker is told
on load whether it gathers, and imports `gather_request` from the wheel it installed, as
every worker imports `run_request`. Until Q458 the page carried the gathering functions as
source, because the wheel then served had been built before they existed; that tied what
the browser ran to the checkout serving the page rather than to the wheel's own ref and
fingerprints. A wheel from before Q458 has no `gather_request`, and the worker refuses it
by name at load.

**The cut is fixed, and the worker count is not in it.** Every phase is cut into
`slices.CUT` units whatever the number of workers, so one worker and four are dealt the
same units and gather the same bytes, and a finished unit has one name however many
workers made it. More units than workers also evens the load: the tournament's schedule
runs from the cheapest pairings to the dearest, so a cut into four leaves the last worker
with every 6,400-node game (B3 measured that unit at 30 s native against the others' few).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from chess_variant_sandbox.tournament import slices


def drive(
    spec: dict[str, Any],
    run: Callable[[str], str] = slices.run_request,
    cut: int = slices.CUT,
) -> dict[str, Any]:
    """The page's coordinator in one process: every unit run here, in order.

    What the browser is measured against, as `slices.measure_in_slices` is B3's. It goes
    through the same texts the page moves -- `slices.gather_request`'s answers and each
    unit's request and reply -- and so differs from the page only in where the strings go.
    """
    answer = json.loads(slices.gather_request(json.dumps({"spec": spec, "state": None}), cut))
    while answer["result"] is None:
        replies: dict[str, list[str]] = {}
        for unit in answer["units"]:
            replies.setdefault(unit["phase"], []).append(run(unit["request"]))
        answer = json.loads(
            slices.gather_request(json.dumps({"state": answer["state"], "replies": replies}), cut)
        )
    result: dict[str, Any] = answer["result"]
    return result
