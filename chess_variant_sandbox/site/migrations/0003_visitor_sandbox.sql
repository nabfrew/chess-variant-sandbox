-- A sandbox seat may be a visitor: neither a person nor a computer player.
--
-- §4 of docs/plan-website.md says a sandbox game counts for nothing and appears in no
-- statistic, and nothing in that description needs an identity. Task 1.4 still wrote
-- every seat as exactly one of the two, so with no way to sign in yet (task 1.3) the
-- site refused every visitor a game at all. The row is what a game in progress is here,
-- so the visitor gets a row with nobody named in their seat.
--
-- Leaderboard mode is exactly as strict as it was. Its result moves a rating table whose
-- rows are people and computer players, so a seat there must still be one of them, and
-- the constraint says so per row rather than being dropped and re-stated in Python.
--
-- Both seats null and the mode sandbox is two visitors, which nothing offers and nothing
-- refuses: one person on one browser is what phase 1 seats, and a second person is
-- phase 3's question with the time controls.

alter table matches drop constraint matches_seat0_is_one_player;
alter table matches add constraint matches_seat0_is_one_player
    check (
        (seat0_user is null) <> (seat0_agent is null)
        or (mode = 'sandbox' and seat0_user is null and seat0_agent is null)
    );

alter table matches drop constraint matches_seat1_is_one_player;
alter table matches add constraint matches_seat1_is_one_player
    check (
        (seat1_user is null) <> (seat1_agent is null)
        or (mode = 'sandbox' and seat1_user is null and seat1_agent is null)
    );
