-- Correspondence play, the match (Q570, `docs/plan-return.md` §1.3, §1.6, §1.7 and §1.9).
--
-- A match between two people is a `matches` row with a person in both seats, which the
-- seat constraints have always allowed. It needs three more columns and one switch.
--
-- **`draw_offer`** is the seat (0 or 1) whose draw offer stands, and null when none does.
-- **`draw_offer_ply`** is how many plies had been played when it was made. An offer lapses
-- when the offerer's opponent moves, which `site/correspondence.py` writes as clearing
-- both; the other person accepting it finishes the match with `reason = 'agreed'`. The
-- two are null together or set together.
--
-- **`times`** is when each move landed, one timestamp per entry of `moves`, for the day
-- the match page shows beside each move between two people (s4 of the mock). Null on
-- every other row: a game against the computer is played in one sitting, and its row
-- keeps only `moved`. Kept beside `moves` the way `depths` is, but without a length
-- check, because a row written before this column has moves and no times.
--
-- **`switches`** holds the site's on/off settings that `/admin` flips, one row a switch
-- that is *on* while the row exists. The first is `correspondence-paused` (§1.7): new
-- games between people are refused while it stands, and games in flight carry on.
-- `set_by` does not cascade, for `suggestions.author`'s reason: the maintainer's account
-- leaving does not unpause anything.
--
-- A cancelled match (either person, before both have moved) is `finished` with a null
-- `result` and `reason = 'cancelled'`; no column is needed, and every reading that counts
-- a result already wants one.

alter table matches add column draw_offer smallint check (draw_offer in (0, 1));
alter table matches add column draw_offer_ply integer;
alter table matches add column times jsonb;
alter table matches add constraint matches_draw_offer_whole
    check ((draw_offer is null) = (draw_offer_ply is null));

create table switches (
    name    text primary key,
    set     timestamptz not null default now(),
    set_by  bigint references users (id)
);
