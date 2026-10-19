-- Proven positions on the site (R9b, Q605, `docs/plan-website.md` *Proven positions*).
--
-- A proven position is a position from one of the library's recorded games that the
-- computer proved won for the side to move (`runs/puzzles/<hash>.json`, R9a). It is
-- reached by replaying a move prefix, so there is no new position format, and it is
-- offered two ways: *convert it*, a real game from the position against the strongest
-- computer player, and *find the move*, a lookup in the stored scores.
--
-- **A try at converting one is a match like any other, started from its prefix.**
-- `matches.moves` holds the whole list from the opening, prefix first, so every reader
-- that replays a row (the board, `finishes`, the export) reaches the right position with
-- no change. Three columns say where the prefix ends:
--
-- - `source` is `'puzzle'` (the column has held `'local-app'` since `0001`), which keeps
--   the row out of every reading that already filters on it;
-- - `puzzle` is the position's id in its record (`seed-game-ply`), on the row's game;
-- - `puzzle_ply` is the prefix's length, so a take-back can never reach into it and a
--   restart goes back to the position rather than the opening.
--
-- The checks hold a try to what it is: the two columns together or not at all, practice
-- only (it counts for nothing and moves no rating), and always with the `puzzle` source.
--
-- **`proven_marks`** is a person's own solved marks, one row a person a position a way:
-- `converted` when a try's game ended in their win (written in the finish's own
-- transaction, `site/proven.py::converted`), `found` when they picked the move the proof
-- names (`site/proven.py::find`). Read by that person's page alone: no count across
-- people is read from it anywhere (test 6 of `docs/reference/no-dark-patterns.md`), a
-- failed try writes nothing, and nothing in it expires. `match` names the try a
-- `converted` mark came from, and is null for `found` (and for a mark whose match row
-- went, which nothing does today). Leaving deletes a person's marks
-- (`admin.forget`); the user row stays, as `0005_leaving.sql` keeps it.

alter table matches add column puzzle text;
alter table matches add column puzzle_ply integer check (puzzle_ply >= 0);
alter table matches add constraint matches_puzzle_whole
    check ((puzzle is null) = (puzzle_ply is null));
alter table matches add constraint matches_puzzle_is_practice
    check (puzzle is null or (mode = 'sandbox' and source = 'puzzle'));

create table proven_marks (
    person   bigint not null references users (id) on delete cascade,
    game     text not null references games (hash) on delete cascade,
    position text not null check (position <> '' and length(position) <= 64),
    way      text not null check (way in ('converted', 'found')),
    match    text references matches (id) on delete set null,
    marked   timestamptz not null default now(),
    primary key (person, game, position, way)
);
