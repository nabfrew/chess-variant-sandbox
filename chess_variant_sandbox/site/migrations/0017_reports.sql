-- Report a game (Q503d, `docs/plan-game-pipeline.md` §3.2 and §5).
--
-- A signed-in person reports a game from its page, and the maintainer reads every report
-- on `/admin` with the game's existing kill switch beside it. A report files a row and does
-- nothing else: it takes nothing off the listing, mails nobody and moves no count. Killing
-- stays the admin's act (`admin.kill_game`), so a report is a thing to look at and never a
-- vote that hides a game.
--
-- **One report a person a game** is the primary key, so it is true of the table rather
-- than of the one writer. `quota.REPORTS_A_DAY` is the other limit, counted over
-- `created`, which is what `reports_by_reporter` is for.
--
-- `reason` is closed to the three words the page offers, the plan's own list: *spam*,
-- *offensive*, *a copy* (§3.2). `note` is optional and short, the same rule
-- `suggestions.body` follows: the cap is a check constraint so that it holds of the table.
--
-- `reporter` does not cascade, which is `suggestions.author`'s rule: leaving empties the
-- person out of their rows rather than removing them (`0005_leaving.sql`), and a report by
-- somebody who has gone still says what they saw. `game` does cascade, though nothing
-- removes a games row today (`0010_deleted.sql` keeps the row behind a null spec).

create table reports (
    reporter bigint not null references users (id),
    game     text not null references games (hash) on delete cascade,
    reason   text not null check (reason in ('spam', 'offensive', 'copy')),
    note     text check (note is null or (note <> '' and length(note) <= 500)),
    created  timestamptz not null default now(),
    primary key (reporter, game)
);

-- The rolling day `quota.reports_today` counts over.
create index reports_by_reporter on reports (reporter, created);
