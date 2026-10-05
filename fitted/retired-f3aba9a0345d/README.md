# Procedure `f3aba9a0345d`: retired

Stage 30B's second fit procedure. It replaced the `material` column of the design matrix
with one column per non-royal piece type, so the fitted coefficients were the piece
values, and it existed to answer one question with a measurement: **does letting a player
say what the pieces are worth make it stronger?**

It does not. Accepted on 5 of 41 rulesets against the current procedure's 10, and beaten
in the direct head-to-head on 4 of the 5 where both arms were accepted. Training loss
improved on 37 of 41 and the play fell on 27, which is overfitting rather than a bad
implementation. The full finding, including every fitted table beside the derived one, is
in `docs/reference/fitted-piece-values.md`.

**The procedure was deleted, as its plan said it would be if it lost.** Nothing can
produce these files any more, and nothing reads them: `load_all` globs `fitted/*.json` and
does not descend, so they are evidence rather than players. They are kept here because a
stage that only planned for its own success would have deleted them, and a refusal is a
result. `values` in each file is the fitted valuation, on the derived table's scale.

The substrate a valuation rides on survives -- `Heuristic.values`, `PieceValues`, the
ninth `AgentSpec` slot, and the browser's per-piece sliders -- because an opponent whose
piece values a person set is *legible*, which is an argument that has nothing to do with
strength.
