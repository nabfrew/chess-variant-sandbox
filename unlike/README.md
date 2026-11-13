# unlike/

The games designed for the search for strong games unlike chess (R-S10b). The design, the
rules a game enters by, what *unlike* means and what each reading predicts are all in
`docs/reference/unlike-search.md`, registered before any game here existed.

- `unlike/<name>.json`: a designed game, its hash written into the record (§9) before it is
  measured.
- `unlike/sessions/<name>.md`: its design reason, break attempts, design log and blind read
  (the record's §2).

These games are **not the library**. They test the score; none is in `rulesets/` or
`archive/`, and whether any ever joins the library is a separate decision. `corpus.find`
reads them on their own rung, `unlike`, measured as a library game is (its seeds, roster and
waiting schedule) and counted by nothing the library counts: not `corpus_hashes`, the site,
or `measure` with no argument. `tests/test_unlike_rung.py` pins that.

The search's published games are not here: they go into `labelled/` as additions to the
labelled set (`docs/reference/labelled-set.md` §3.3).
