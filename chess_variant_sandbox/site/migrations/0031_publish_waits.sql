-- *Publish* waits for the site's check instead of waiting to be pressed (Q742,
-- `docs/plan-game-pipeline.md` §7, `site/drafts.py`).
--
-- Until Q742 the button stayed shut until the score measured on the designer's computer
-- had been walked (tier one) and spot-checked (tier two). Now it opens once the browser
-- has finished and sent the score, and a press while the site is still checking it is
-- recorded here; the runner that finishes the check puts the game on trial
-- (`drafts.settle`). Nothing goes live unchecked: the press only asks.
--
-- **`games.publish_asked`** is when the designer pressed *Publish* on a draft whose score
-- the site was still checking. Null on every row that is not waiting. The terms and their
-- version are stamped at the press, as before (`0029_terms.sql`): the tick is taken then.
-- The runner clears it when the game goes on trial, or when the check fails, the account
-- is flagged, the game is taken off the site or a gate the press passed no longer holds.
--
-- **`games.publish_stopped`** is why the last waiting publish was called off, in the
-- sentence its designer reads on the *Publish* panel. Cleared by the next press and by
-- going on trial.

alter table games add column publish_asked timestamptz;
alter table games add column publish_stopped text;
