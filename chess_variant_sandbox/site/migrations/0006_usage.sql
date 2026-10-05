-- How the site is used, counted by the hour and never by the person.
--
-- `docs/plan-usage.md` §2. Three tables, written by `site/usage.py`. No column names a
-- person, holds an address or reads a cookie, which is what the privacy notice promises
-- and what `tests/test_site.py`'s `STORE` pins: a usage table with a user column fails
-- there before it fails a reader.

-- One row per hour per route per outcome. `route` is the template from `site/routes.py`
-- with its method, so a game hash or a match id never enters the table. `variant` is what
-- the template cannot say and the cost depends on: a session action, an agent's budget,
-- a `show` kind. `kind` is the exception class for a refusal, empty for a 200.

create table usage_hours (
    hour timestamptz not null,
    route text not null,
    variant text not null default '',
    status smallint not null,
    kind text not null default '',
    who text not null check (who in ('member', 'visitor')),
    count integer not null,
    cpu double precision not null,
    wall double precision not null,
    wall_max double precision not null,
    primary key (hour, route, variant, status, kind, who)
);

-- One row per hour per route, about signed-in people as numbers. The ids that produce it
-- live in memory for the hour and are never written. `peak` is the most requests one
-- person made; for *show me* it is the rolling hour's, read off `quota.Presses`. The six
-- buckets count people by how many requests they made. `partial` marks an hour the
-- process did not see whole, whose distinct counts are then an estimate.

create table usage_people (
    hour timestamptz not null,
    route text not null,
    variant text not null default '',
    people integer not null,
    refused integer not null,
    peak integer not null,
    one integer not null,
    two_to_five integer not null,
    six_to_ten integer not null,
    eleven_to_twenty integer not null,
    twenty_one_to_fifty integer not null,
    over_fifty integer not null,
    partial boolean not null default false,
    primary key (hour, route, variant)
);

-- One row every five minutes saying what the machine had left. A column is null where
-- the file it is read from does not exist, as on a development machine.

create table usage_samples (
    at timestamptz primary key,
    cpu_busy real,
    app_cpu double precision,
    load1 real,
    mem_available integer,
    app_memory integer,
    disk_free integer,
    in_flight_max integer,
    strong_seats_max integer
);
