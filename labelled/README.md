# The labelled set

A set of games each labelled *good* or *bad* (or *null*, a control) by a source that never
reads the score, so that the score can be tested against labels that cannot follow it. The
design, its rules and the reading it is for are `docs/reference/labelled-set.md` (R-S2a);
read that first. Nothing here is part of the library: no game in this directory is in
`rulesets/` or `archive/`, the site's listing, the corpus counts or `measure`'s sweep.

| path | holds |
|---|---|
| `<name>.json` | the spec of a member that is not a library game (new published games; later the bad games and controls) |
| `labels/<name>.json` | one label per member, the 21 library games included (they name their `rulesets/` or `archive/` file) |
| `certificates/<name>.json` | a forced-win certificate, where a proof is one |
| `verify.py` | walks every certificate through the engine and records the ones that hold in `verified.json` |
| `check.py` | re-runs the mechanical part of every argued proof (`proof.checks` in its label) |
| `prove.py` | the searches behind the proofs: forced wins, and every position reachable under a recipe |
| `likeness.py` | places a game in its likeness quarter, against the library as R-S2b found it |
| `build-log.md` | every candidate skipped, cell left empty and game dropped, with its reason |
| `SEAL.json` | the seal (R-S2b2), after which `reading.json` (R-S2c) may follow |

**The rules that bind this directory** (§3.2, held by `tests/test_labelled_set.py`):

- A label, a spec and a certificate are each written by exactly one commit, and never edited
  or removed. A wrong label gets a line in `withdrawn.md` instead.
- No member built for the set is measured or played by this project's players before the
  seal. A game here is measured with `uv run chess-variant-sandbox measure labelled/<name>.json`,
  which gives it the library's four runs and waiting schedule (`corpus`'s `labelled` rung),
  and only by R-S2c.
- A game whose score anyone in the project has seen is never added as bad.
