-- One row per badge earned, and the range a score badge quotes.
--
-- Box E1 of `docs/plan-meta-game.md`, against §2.9. A milestone is a live reading over
-- match rows and reopens when a ladder or the library grows. A badge is kept once
-- earned, which a live reading cannot promise, because three of a badge's inputs change
-- or vanish after the fact: a vote is replaced and `votes` keeps no history, a score
-- goes stale when the weights or the sources move, and the count of available rules
-- grows with each schema version. So the site stores the event itself.
--
-- A row is written once and never updated. No re-measure, re-anchor, changed vote or
-- moved threshold removes one. `rule` is what makes that readable years later: the
-- thresholds in force at the grant travel with the row, so an award still explains
-- itself after the constant moves.
--
-- Three of the four key columns are nullable, and each null means something. `tier` is
-- null for a badge with no series, `holder` for a badge that sits on a library game, and
-- `game` for a badge earned across the library or once per person. Postgres treats nulls
-- as distinct in a unique index by default, which would let the same badge be granted
-- twice, so the constraint is declared `nulls not distinct` -- PostgreSQL 15 and above,
-- which is what the box runs.
--
-- `holder` does not cascade. Leaving empties a person out of their rows rather than
-- removing them (`0005_leaving.sql`), and `evidence` holds ids and figures and never a
-- handle, so a departed person reads as *a former player* on an award as on a match.
-- `game` does cascade: a game that is deleted takes its awards with it.

create table awards (
    kind        text not null,
    tier        integer,
    holder      bigint references users (id),
    game        text references games (hash) on delete cascade,
    earned      timestamptz not null default now(),
    evidence    jsonb not null default '{}'::jsonb,
    rule        jsonb not null default '{}'::jsonb,
    constraint awards_once unique nulls not distinct (kind, tier, holder, game)
);

create index awards_by_holder on awards (holder) where holder is not null;
create index awards_by_game on awards (game) where game is not null;

-- The two reads the pages make: a person's badges on their profile, and a game's badges
-- on its page. Both are partial, because a badge with a null holder is on no profile and
-- a badge with a null game is on no game page.


-- The range a score badge quotes, beside the headline it quotes it with.
--
-- `measurements` keeps `headline`, `state` and `basis`, and the interval lives in each
-- run directory's `index-<weights>.json`. U1 tests a game's whole range inside a vote's
-- transaction, which should not open files, so the loader brings the two ends into the
-- row it already writes. Both are null unless that measurement reads current, exactly as
-- `headline` is: a stale score is not a number and neither are its ends.

alter table measurements add column ci_low double precision;
alter table measurements add column ci_high double precision;
