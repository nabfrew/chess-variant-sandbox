-- The site's whole store, as §3 of docs/plan-website.md lists it.
--
-- Every table lands here rather than one per phase-1 row, because a column added later
-- is a migration and a table added later is a design change. This file is applied once
-- and never edited afterwards; `newgame/site/migrate.py` refuses a changed file.
--
-- Two conventions run through it. A flag that wants a date is a nullable `timestamptz`
-- rather than a boolean, so `revoked` and `nominated` answer *when* for free and read as
-- true by `is not null`. A seat or a player is either a person or a computer player, so
-- it is a nullable user reference beside a nullable agent string with a check that
-- exactly one is filled -- an agent string carries its budget and its objective, which
-- is why it is the identity rather than a row in a table of agents.
--
-- Three of §3's column names collide with Postgres, and the three were settled against a
-- real server (15.4) rather than from memory, because the two that matter fail in
-- opposite ways:
--
--   * `matches.left` is **quoted**. Unquoted it is a syntax error, so a query that
--     forgets the quotes cannot run at all. The plan's word survives and the failure is
--     loud, which is the trade worth making.
--   * `votes.user` is **renamed** to `voter`. Unquoted it is not an error -- `select user`
--     is a working expression that returns the session role -- so a query that forgot the
--     quotes would answer `newgame` for every row and look fine. A name that fails
--     silently is not worth the plan's word.
--   * `votes.text` is **renamed** to `note`. Not a reserved word and legal unquoted; this
--     one is readability only, so that the column does not read as its own type and so
--     that the table's two renamed columns are consistent.
--
-- No schema is named anywhere, so the objects land wherever `search_path` points. The
-- box's site connects as the `newgame` role, which owns `public` there, and a test runs
-- the whole file inside a scratch schema it drops.


-- ------------------------------------------------------------------ people and access

create table users (
    id          bigint generated always as identity primary key,
    handle      text unique,
    email       text not null unique,
    password    text,
    invited_by  bigint references users (id),
    created     timestamptz not null default now(),
    role        text not null default 'member',
    revoked     timestamptz
);

-- `handle` and `password` are null between the invite being accepted and the first
-- login, which is where both are chosen. `password` holds a `hashlib.scrypt` digest with
-- its parameters and salt, never a password, and `email` is in clear because the sign-in
-- link is sent to it -- the privacy notice says both.

create table invites (
    code        text primary key,
    issuer      bigint references users (id),
    uses_left   integer not null,
    expiry      timestamptz,
    created     timestamptz not null default now()
);

-- `issuer` is null for the first invite, which predates every user.

create table sessions (
    token       text primary key,
    user_id     bigint not null references users (id) on delete cascade,
    created     timestamptz not null default now(),
    seen        timestamptz not null default now(),
    expires     timestamptz
);

create index sessions_by_user on sessions (user_id);

-- Not in §3's table until the DDL went looking for it: the cookie has to be checked
-- against something, and "one admin command revokes every session" (1.3) is a delete
-- over these rows, which `sessions_by_user` is the index for.


-- ------------------------------------------------------------------- games and scores

create table games (
    hash        text primary key,
    spec        jsonb not null,
    name        text not null,
    owner       bigint references users (id),
    visibility  text not null default 'private',
    parent      text references games (hash),
    origin      text not null,
    created     timestamptz not null default now(),
    nominated   timestamptz,
    revoked     timestamptz
);

-- The mechanics hash is the key, so the same rules uploaded twice are one row and the
-- first uploader keeps the credit; the primary key is the by-hash index §3 asks for.
-- `owner` is null for a corpus game loaded before any user exists. `revoked` is 1.8's
-- per-game kill switch, which overrides `visibility` rather than borrowing a value from
-- it, because visibility is the owner's answer and the kill switch is not.

create table measurements (
    hash        text primary key references games (hash) on delete cascade,
    directory   text not null,
    state       text not null,
    basis       jsonb,
    headline    double precision,
    ladder      jsonb,
    top         text,
    top_rating  double precision,
    updated     timestamptz not null default now()
);

-- A view over `runs/`, rebuilt by the loader from the files, which stay the truth.
-- `directory` is `<hash>-<seed>` under the runs root, `basis` is what
-- `newgame/measurement.py` stamps and `state` is what `Staleness` makes of it.
-- `ladder` is the whole `runs/opponents/<hash>.json`, which seeds the game's rating
-- table once. `top` and `top_rating` are that ladder's top rung as loaded -- the seed's
-- top, never the live one, which §5 reads off `ratings` and which human play moves.

create table ratings (
    id            bigint generated always as identity primary key,
    game          text not null references games (hash) on delete cascade,
    player_user   bigint references users (id) on delete cascade,
    player_agent  text,
    rating        double precision not null,
    deviation     double precision not null,
    volatility    double precision not null,
    matches       integer not null default 0,
    established   boolean not null default false,
    updated       timestamptz not null default now(),
    constraint ratings_player_is_one_or_the_other
        check ((player_user is null) <> (player_agent is null))
);

create unique index ratings_one_row_per_person
    on ratings (game, player_user) where player_user is not null;
create unique index ratings_one_row_per_agent
    on ratings (game, player_agent) where player_agent is not null;
create index ratings_by_game on ratings (game, rating desc);

-- One Glicko-2 table per game with people and computer players on it. The two partial
-- unique indexes together are §3's "one row per player per game", each covering the half
-- of the table its column is filled on. `ratings_by_game` is the game's table in display
-- order and the top query's index. The table is a fold over `matches` from its seed, so
-- these rows are a cache that replaying the match rows reproduces.


-- --------------------------------------------------------------------------- playing

create table matches (
    id                      text primary key,
    game                    text not null references games (hash) on delete cascade,
    mode                    text not null check (mode in ('sandbox', 'leaderboard')),
    seat0_user              bigint references users (id),
    seat0_agent             text,
    seat1_user              bigint references users (id),
    seat1_agent             text,
    seed                    bigint not null,
    moves                   jsonb not null default '[]'::jsonb,
    result                  double precision,
    reason                  text,
    "left"                  boolean not null default false,
    clock                   jsonb,
    started                 timestamptz not null default now(),
    moved                   timestamptz,
    finished                timestamptz,
    source                  text,
    top                     boolean not null default false,
    seat0_rating_before     double precision,
    seat0_deviation_before  double precision,
    seat0_rating_after      double precision,
    seat0_deviation_after   double precision,
    seat1_rating_before     double precision,
    seat1_deviation_before  double precision,
    seat1_rating_after      double precision,
    seat1_deviation_after   double precision,
    constraint matches_seat0_is_one_player
        check ((seat0_user is null) <> (seat0_agent is null)),
    constraint matches_seat1_is_one_player
        check ((seat1_user is null) <> (seat1_agent is null))
);

create index matches_by_game on matches (game, finished);
create index matches_by_seat0 on matches (seat0_user);
create index matches_by_seat1 on matches (seat1_user);

-- `id` is the session string the page already carries, the one `web.human.HumanGame`
-- calls `session`, so reopening a tab is a lookup. In the local app that string is
-- `secrets.token_urlsafe(9)` and unguessability is the whole of the access control; on
-- the site the seat columns and 1.3's cookie can say who may look, so 1.4 should decide
-- outright whether this column stays a secret or becomes a name the page may print. The
-- column holds either.
--
-- `result` is seat 0's score -- 1, 0.5 or 0 -- and is null while the match runs. `moved` is when the last move landed, which
-- is what 1.4's idle sweep resigns on; `clock` waits for live play and is here because a
-- later column is a migration. `source` marks a row loaded from the local app, which no
-- leaderboard counts. `top` is whether the computer seat was the top of the table when
-- the match started, and is what the *Unbeaten* tag and a person's record read; a match
-- between two people leaves it false.
--
-- `matches_by_game` is also the fold's index: §5 replays a game's leaderboard rows in
-- finish order. A person's matches are a union over the two seat indexes.

create table votes (
    voter       bigint not null references users (id) on delete cascade,
    game        text not null references games (hash) on delete cascade,
    stars       smallint not null check (stars between 1 and 5),
    tags        text[] not null default '{}',
    note        text,
    updated     timestamptz not null default now(),
    primary key (voter, game)
);

-- One row per person per game, replaced on a re-vote, which the primary key enforces.
-- `voter` and `note` are §3's `user` and `text` renamed; the header says why.


-- ------------------------------------------------------------------------- the queue

create table jobs (
    id          bigint generated always as identity primary key,
    kind        text not null,
    hash        text references games (hash) on delete cascade,
    requester   bigint references users (id),
    state       text not null,
    worker      text,
    fingerprint text,
    created     timestamptz not null default now(),
    claimed     timestamptz,
    finished    timestamptz,
    error       text,
    log         text
);

create index jobs_queue on jobs (state, created);

-- A worker claims a row with one atomic update, which `jobs_queue` is the index for, and
-- carries its `_core.BUILD_FINGERPRINT` in `fingerprint`; a result whose fingerprint
-- differs from the server's build is refused. `log` is the tail, not the log.
