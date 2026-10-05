-- A save no longer lists a game; *Publish* does (Q503a, `docs/plan-game-pipeline.md` §5).
--
-- Until now `drafts.keep` wrote a designed row `public` when the playability check passed
-- and `unlisted` when it did not, so `visibility` carried two facts at once: whether the
-- rules passed the check, and whether the game was listed. The pipeline's five levels pull
-- them apart. A save keeps a Draft, unlisted whatever the check says, and the check's
-- verdict is a column of its own. A designed game goes on trial only when its designer
-- presses *Publish*, which needs the check passed and a spot-checked reading attached
-- (answer Q5: publishing waits for the checking step, Q503e).
--
-- **`screen_passed`** is the check's verdict at the latest save. For every row already
-- here it is copied off the visibility that encoded it: a designed row was `public`
-- exactly when its check passed. That is the old reading moved into its own column, not
-- a new one, which is why no *passes the screen* badge moves (`site/badges.py` reads the
-- column now). A corpus or archive row was never screened on the site and reads false;
-- nothing asks it.
--
-- **`published`** is when the designer pressed *Publish*, null for every row here. The
-- day's cap (`quota.PUBLISHES_A_DAY`) counts it.
--
-- **Existing public designed rows go back to Draft** (§6's third *needs decision*, option
-- A, decided 2026-09-28). They carry no reading, and a game on trial without one is the
-- thing the plan exists to stop. `unlisted`, so every link to them still opens and plays;
-- the designer's profile says why (`site/drafts.returned`). Nothing else is stamped:
-- no `published`, no reading, no nomination, and `created`, `owner` and the kill switch
-- stay as they were.

alter table games add column screen_passed boolean not null default false;
alter table games add column published timestamptz;

update games set screen_passed = true
    where origin = 'designed' and visibility = 'public';

update games set visibility = 'unlisted'
    where origin = 'designed' and visibility = 'public';
