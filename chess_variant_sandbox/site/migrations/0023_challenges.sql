-- Correspondence play, finding an opponent (Q571, `docs/plan-return.md` §1.4 and §1.9).
--
-- **`challenges`** is one challenge: a member asking another to play one game of one
-- game's rules, by correspondence. It is named or it is a link:
--
-- - a **named** challenge has `challenged` from the start and no `token`;
-- - a **link** has a `token` (the `/c/<token>` address) and no `challenged` until a
--   member accepts it, when the acceptor is written there. A link works once.
--
-- `seat` is the challenger's: 0 moves first, 1 moves second, null is *either*, drawn
-- from the match's seed when it starts. Whether the game is rated is not a column: it is
-- the game's (`correspondence.start` decides it from the ladder), never a choice.
--
-- `answer` is null while the challenge waits, which is for as long as it takes: a
-- challenge never expires (test 2 of `docs/plan-meta-game.md`). `accepted` writes
-- `match`, the row the challenge became; `declined` is the challenged person's answer,
-- `withdrawn` the challenger's, or what leaving the site or being revoked does to every
-- challenge a person sent or was sent. `answered` is when.
--
-- `rematch_of` is the finished game between the same two people this one was offered
-- from (*Rematch* in its banner), with the seats swapped; null on every other challenge.
--
-- **`open_to`** is a member's *open to a game of this* box, one row a ticked game. Off by
-- default, which is the absence of a row, and it never lapses (test 1).
--
-- No cascade from `users`: an account is never deleted, only revoked or forgotten, and
-- leaving withdraws and unticks in the same transaction. A game's rows go with the game.

create table challenges (
    id          bigint generated always as identity primary key,
    challenger  bigint not null references users (id),
    challenged  bigint references users (id),
    token       text unique,
    game        text not null references games (hash) on delete cascade,
    seat        smallint check (seat in (0, 1)),
    rematch_of  text references matches (id) on delete set null,
    made        timestamptz not null default now(),
    answer      text check (answer in ('accepted', 'declined', 'withdrawn')),
    answered    timestamptz,
    match       text references matches (id) on delete set null,
    constraint challenges_named_or_link check (challenged is not null or token is not null),
    constraint challenges_not_oneself check (challenged is null or challenged <> challenger),
    constraint challenges_answered_whole check ((answer is null) = (answered is null))
);

create index challenges_waiting_for on challenges (challenged) where answer is null;
create index challenges_sent_by on challenges (challenger, made);

create table open_to (
    person  bigint not null references users (id),
    game    text not null references games (hash) on delete cascade,
    since   timestamptz not null default now(),
    primary key (person, game)
);

create index open_to_game on open_to (game);
