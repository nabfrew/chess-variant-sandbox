-- How near two of the store's games are, and what differs between them.
--
-- `docs/plan-game-graph.md` G7. `spec/relate.py` reads two specs rule by rule and says
-- what differs; up to now the site kept those readings in `runs/graph/pairs.json`, a file
-- keyed by hash and nothing else. A file cannot say who may see a game, and phase 2 puts
-- fifty drafts in the library that not everybody may open. This table is the same reading
-- in the store, where `games` can be joined onto it and the question *may this person be
-- shown this link* has an answer.
--
-- **Both ends, in sort order, one row.** A link is symmetric, so `a < b` is enforced
-- rather than remembered and the pair is the primary key. A reader asks
-- `where a = :hash or b = :hash` and takes the other end.
--
-- **Every pair, not only the near ones.** `units` is written for all of them, because a
-- game far from everything is shown its nearest few and that ranking needs every pair.
-- The reading -- `kind`, `kinds`, `unit`, `sentence`, `differences` -- is written only
-- within three rules, which is `web/library.py`'s `NEAR` and the threshold the plan sets:
-- past it a list of nine differences is a ranking of how unlike everything else a game
-- is rather than a reading of it. So a far pair is one integer and two hashes.
--
-- **`relate` is the version the row was read under** (`spec/relate.py`'s
-- `RELATE_VERSION`). A row stamped with anything else is recomputed rather than trusted,
-- which is what lets the reading change without a migration. It is on the row rather than
-- on the table because a load fills what is missing and leaves the rest alone.
--
-- **The cascade is the second line, not the first.** Deleting a game's rules does not
-- delete its row (`0010_deleted.sql`), so `admin.delete_game` removes the link rows
-- itself -- a link spells out rule differences, and the user's decision of 2026-09-18 is
-- that the rules go. The foreign keys are still `on delete cascade`, so a store where a
-- `games` row really does go leaves no link naming a hash that is not there.

create table if not exists links (
    -- The two games, always in sort order, so one pair is one row whichever way it is
    -- asked about.
    a           text not null references games (hash) on delete cascade,
    b           text not null references games (hash) on delete cascade,
    units       integer not null,
    -- The reading, within three rules; empty or null past it. `kind` is the first
    -- difference's kind, which is the whole kind of a one-rule link and the one a mixed
    -- link is filed under; `kinds` is every kind on it, so a filter never guesses.
    kind        text not null default '',
    kinds       text[] not null default '{}',
    unit        text not null default '',
    sentence    text not null default '',
    differences jsonb,
    relate      text not null,
    computed    timestamptz not null default now(),
    primary key (a, b),
    constraint links_are_ordered check (a < b)
);

-- The one read: this game's links, nearest first. It is asked from either end, so both
-- columns carry the index and both carry `units` beside them for the ordering.

create index if not exists links_by_a on links (a, units);
create index if not exists links_by_b on links (b, units);
