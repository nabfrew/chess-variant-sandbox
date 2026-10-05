-- What `newgame site forget` needs, which is the privacy notice's own promise.
--
-- Task 1.8. The notice at `/privacy` says leaving deletes the email address and the
-- password hash and turns the name on every match into *a former player*. The rows the
-- matches point at have to stay -- each game's ladder is a fold over its matches from
-- the first, so deleting a user would move other people's ratings -- so leaving empties
-- the person out of the row rather than removing it.
--
-- `email` is `not null` today, and a sentinel address per departed person would be the
-- same address stored twice under a different name. Dropping the constraint is what lets
-- the column actually be empty. The unique index stays and allows any number of nulls,
-- which is the behaviour wanted: two people who left are not the same person.
--
-- The kill switches need nothing new. `users.revoked` and `games.revoked` are both in
-- `0001_store.sql` and both already honoured -- `identity.sign_in` and `identity.resolve`
-- filter on the first, and 1.8 adds the reading and the page for the second.

alter table users alter column email drop not null;

-- When they left, which is not when they were killed. A killed account is a person the
-- maintainer shut out and can let back in; a forgotten one is a person who asked to go
-- and whose email and password are gone, so nothing can let them back in. One column
-- each, so the admin page can say which happened rather than guessing from empty fields.

alter table users add column forgotten timestamptz;
