-- Resting: a trial game nobody plays leaves the listing (Q503i, `docs/plan-game-pipeline.md`
-- §2.2's last row), and the Baseline's moves are remembered (Q503n, §2.1 and §3.2).
--
-- **`rested`** is when the sweep (`site/resting.py`, run by `chess-variant-sandbox site
-- load`) took a trial game off the listing: 90 days on trial (`quota.RESTING_AFTER`) with
-- no game finished on it by anybody but its designer in the last 60
-- (`quota.RESTING_QUIET`). The row keeps `visibility = 'public'`, so its page and link work
-- as before; `stages.stage` reads the stamp as **Resting**, ahead of the visibility. A
-- shortlisted game never rests: the sweep leaves a nominated row alone. Null for every row
-- here, so nothing rests until the first sweep after this migration.
--
-- **`woken`** is when its designer published a resting game again, which clears `rested`
-- and restarts `published`. The plan allows that once (*"the designer may publish it again
-- once"*), so a row that rests a second time with `woken` set stays resting.
--
-- **`level_moves`** is one row per time `site load` found a library game had moved between
-- `archive/` (Library) and `rulesets/` (Baseline), which only the maintainer does
-- (`docs/reference/baseline-procedure.md`). `reason` is the first sentence of the spec's
-- `provenance.history` when the move changed it -- what `chess-variant-sandbox promote
-- --reason` writes -- and empty when it did not. The game page's *Where it stands* and
-- *What's new* read it; no route writes it. `game` cascades, though nothing removes a games
-- row (`0010_deleted.sql` keeps it behind a null spec).

alter table games add column rested timestamptz;
alter table games add column woken timestamptz;

create table level_moves (
    id      bigserial primary key,
    game    text not null references games (hash) on delete cascade,
    was     text not null check (was in ('archive', 'corpus')),
    became  text not null check (became in ('archive', 'corpus')),
    reason  text not null default '',
    moved   timestamptz not null default now(),
    check (was <> became)
);

create index level_moves_by_game on level_moves (game, moved);
create index level_moves_by_moved on level_moves (moved);
