-- Provisional results: a score a visitor's browser measured, which the server has checked
-- and has not measured itself.
--
-- Q312, the first row of `docs/plan-client-compute.md` §8 after the probe. The browser
-- plays and scores a draft in web workers; the server walks every game for legal play
-- (tier one), later replays a sample of them byte for byte (tier two), and on request
-- measures the game itself (tier three). Only tier three's number is official: it is an
-- ordinary `measurements` row read off an ordinary `canonical.json`. Everything before it
-- lives here, in a table of its own, so that nothing which ranks or awards -- the index
-- sort, *Unbeaten*, the score badges, nomination -- can read a provisional number by
-- accident. None of them knows this table exists.
--
-- **The five states are §5's.** `legal` is tier one passed, and only the submitter sees
-- it. `spot-checked` is tier two passed, and everybody who can see the game sees it,
-- labelled, by the user's answer A of 2026-09-18, kept after T2 on 2026-09-25.
-- `confirmed` is tier three's own measurement landing, after which the official number
-- is what shows. `mismatch` is a check that failed. `superseded` is a newer result for
-- the same game, or the game going. A result is shown only in the first two, and at most
-- one result a game is in them, which the partial unique index holds.
--
-- **The basis is the server's stamp, and it says a browser played the games.**
-- `measurement.Basis` has `played` and `checked` for that. The checks below tie the
-- stamp to the state, so a row cannot say `spot-checked` while its stamp says only that
-- the moves were legal, and the page's label cannot outrun what the server did. Each
-- reads an absent key as `''` through `coalesce`, because a check whose expression is
-- null passes, and an absent key is exactly the stamp these checks exist to refuse.
--
-- **The records live under `runs/provisional/`, never under `runs/<hash>-<seed>/`**, and
-- `directory` is where, relative to the runs root, or null once they are deleted (on
-- `confirmed` and `superseded`). A `mismatch` keeps its records: they are what tells a
-- core bug from a forged submission.
--
-- `submitter` does not cascade, which is `suggestions.author`'s rule: leaving empties the
-- person out of their row rather than removing it (`0005_leaving.sql`). `job` is the
-- queue row that last moved the result, null until the tiers' jobs exist.

create table provisional (
    id          bigint generated always as identity primary key,
    game        text not null references games (hash) on delete cascade,
    submitter   bigint not null references users (id),
    basis       jsonb not null,
    headline    double precision,
    viable      boolean,
    state       text not null,
    job         bigint references jobs (id) on delete set null,
    directory   text,
    created     timestamptz not null default now(),
    updated     timestamptz not null default now(),
    constraint provisional_state_is_one_of_five check (
        state in ('legal', 'spot-checked', 'confirmed', 'mismatch', 'superseded')
    ),
    constraint provisional_played_in_a_browser check (
        coalesce(basis ->> 'played', '') = 'browser'
    ),
    constraint provisional_legal_is_stamped_legal check (
        state <> 'legal' or coalesce(basis ->> 'checked', '') = 'legal'
    ),
    constraint provisional_spot_checked_names_its_sample check (
        state <> 'spot-checked'
        or coalesce(basis ->> 'checked', '') ~ '^sample:[1-9][0-9]*$'
    ),
    constraint provisional_records_under_provisional check (
        directory is null or directory ~ '^provisional/[0-9]+$'
    )
);

-- The one standing result a game has, which is what every reader asks for.
create unique index provisional_one_standing on provisional (game)
    where state in ('legal', 'spot-checked');

-- A game's history, newest last, for the admin log and for superseding.
create index provisional_by_game on provisional (game, created);
