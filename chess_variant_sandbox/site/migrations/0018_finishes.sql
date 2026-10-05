-- Every game a person finished, move by move, so a win can be replayed (Q502a).
--
-- `docs/reference/general-player.md` §4.2 and its E6: the only strength check that is not
-- the ladder measuring itself is a person beating it, and a loss to a person that cannot
-- be replayed cannot be studied. A leaderboard match already keeps its moves and its
-- verdict for good -- it refuses a take-back and a restart -- but a sandbox match does
-- not: `site/matches.py` truncates `moves` on a take-back and empties it on a restart, and
-- `_reopen` clears the verdict. So a sandbox win followed by *play again* was gone. This
-- table is written by the one call that writes a verdict (`matches._finish`), in the same
-- transaction, and nothing updates or deletes a row of it afterwards.
--
-- **One row per finish, not per match.** A sandbox match taken back past its end and
-- finished again is two games that both ended, and both are kept: `ordinal` counts them,
-- 1 for the match's first. A leaderboard match finishes once, so its row is `ordinal` 1
-- and repeats what the match row holds; the copy is one short integer array, and it
-- buys a reader one table for every finished human game rather than a union with a rule
-- about which half is authoritative.
--
-- **Only games somebody sat at.** A watch -- a computer player in both seats -- is not
-- written: nobody played it, and the tournament runner is where machine games live.
--
-- **Nothing about who played is copied.** The row names its match and nothing else: who
-- sat where, the game, the seed and the mode stay on the match row, which is where they
-- were already kept, so leaving (`0005_leaving.sql`) still empties the one place a name
-- is. A visitor's seat on the match row is empty, so a visitor's finish is a move list,
-- a result and a time, attached to nobody.
--
-- `result` is seat 0's score, as `matches.result` is. `reason` is the ending's word: the
-- engine's for an ending on the board, `resigned` for one it cannot see. `takebacks` is
-- how many times the person took moves back on the way (below), always 0 on a
-- leaderboard row, so a win found with the undo button is not mistaken for one played
-- straight. `moves` is the list `matches.moves` held at the finish, which replays through
-- the engine exactly as a match does (`site/finishes.py::replay`).

create table finishes (
    match      text not null references matches (id) on delete cascade,
    ordinal    integer not null check (ordinal >= 1),
    moves      jsonb not null check (jsonb_typeof(moves) = 'array'),
    result     double precision not null check (result in (0, 0.5, 1)),
    reason     text not null,
    takebacks  integer not null check (takebacks >= 0),
    finished   timestamptz not null default now(),
    primary key (match, ordinal)
);

-- Take-backs since the board was last at the opening position: `matches.undo` adds one,
-- `matches.restart` puts it back to 0, and a finish copies it. Only a sandbox match can
-- take anything back, so a leaderboard match keeps the default.

alter table matches add column takebacks integer not null default 0
    check (takebacks >= 0);
