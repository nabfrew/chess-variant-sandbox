-- The round: the members' vote, then the maintainer's confirm or veto (Q503h,
-- `docs/plan-game-pipeline.md` §2.2 *The round, in order*, §3.3 and §3.4).
--
-- **`rounds`** is one row a month, keyed by the 1st of the month it is the round of.
-- `opens` is when the sweep (`site/rounds.py`, run by `site sweep`, `site load` and the
-- round's own pages) opened it and fixed its entrants; `closes` is the 1st plus seven days
-- (`quota.ROUND_VOTING`), whenever it opened; `closed` is when the sweep tallied it. A
-- round that opens with no entrant is still a row, so the month is known to have been
-- looked at.
--
-- **`round_entrants`** is one row per game in a round. `nominated` is the nomination it
-- entered on, which breaks a tie in the order the four places go (§2.2 step 5). A game
-- that passed a round and was neither confirmed nor vetoed before the next one opened is
-- carried into it without a new vote: `carried` names the round whose vote it keeps, and
-- `yes` and `no` are copied from there. For every other entrant `yes` and `no` are null
-- until the close, so no reader can show a running count (§3.3).
--
-- **`round_votes`** is one yes or no a member, a game, a round, replaced until the close.
-- `voted` is when it was last cast (the plan's "cast", a reserved word in SQL).
--
-- **`round_decisions`** is what became of an entrant: `voted-down` (written by the close,
-- its reason the tally), `confirmed` or `vetoed` (the maintainer's, from `/admin`, each
-- with its reason). A voted-down or vetoed game goes back to On trial and may be
-- nominated again `quota.DECLINED_WAIT` (60 days) after `decided`. A confirmed game stays
-- shortlisted until the maintainer's `archive` commit is deployed and `site load` finds
-- its spec in `archive/`; the loader then takes the row over as a Library game, keeping
-- its designer, and writes a `level_moves` row from `designed` (below).
--
-- **`level_moves.was`** gains `designed`: the one move a player-made game makes into the
-- Library, recorded by the loader, which stays the only writer of an origin and of
-- `level_moves` (`tests/test_site_resting.py`).

create table rounds (
    day     date primary key,
    opens   timestamptz not null,
    closes  timestamptz not null,
    closed  timestamptz,
    check (extract(day from day) = 1),
    check (closes > opens)
);

create table round_entrants (
    round      date not null references rounds (day) on delete cascade,
    game       text not null references games (hash) on delete cascade,
    nominated  timestamptz not null,
    carried    date references rounds (day),
    yes        integer check (yes >= 0),
    no         integer check (no >= 0),
    primary key (round, game)
);

create index round_entrants_by_game on round_entrants (game, round);

create table round_votes (
    round   date not null,
    game    text not null,
    person  bigint not null references users (id) on delete cascade,
    yes     boolean not null,
    voted   timestamptz not null default now(),
    primary key (round, game, person),
    foreign key (round, game) references round_entrants (round, game) on delete cascade
);

create table round_decisions (
    round     date not null,
    game      text not null,
    decision  text not null check (decision in ('confirmed', 'vetoed', 'voted-down')),
    reason    text not null default '',
    decided   timestamptz not null default now(),
    decider   bigint references users (id) on delete set null,
    primary key (round, game),
    foreign key (round, game) references round_entrants (round, game) on delete cascade
);

create index round_decisions_by_game on round_decisions (game, decided);

alter table level_moves drop constraint level_moves_was_check;
alter table level_moves add constraint level_moves_was_check
    check (was in ('archive', 'corpus', 'designed'));
