-- A game deleted outright: the rules go, the row stays as a placeholder.
--
-- The user, 2026-09-18: *"Keep it as 'retired', but with a button available to the game's
-- creator and admin to delete completely. The game rules are then actually deleted. If it
-- games played are entangled in other statistics and need to reference it, reference it
-- simply as 'deleted game'."*
--
-- The row cannot go. Every reference to a game in this store is a foreign key with
-- `on delete cascade` -- `matches.game`, `ratings.game`, `votes.game`,
-- `measurements.hash`, `awards.game`, `jobs.hash` -- so deleting the row would delete
-- somebody's matches, their rating table and their votes along with the rules. The
-- alternative shape, nullable references, would make the hash unusable as a join key:
-- the rows that belonged to one game would stop belonging to each other, and *the games
-- you played on it* would become uncountable. So the hash stays and the spec goes, and
-- every count over these tables reads the same number after a delete as before it.
--
-- `deleted` is a third column beside `revoked` and `retired` for the reason those two
-- are separate: an admin's kill switch, a deploy's reading of the tree and an
-- irreversible delete are three acts, and one shared column would let any of them
-- overwrite another. It is the strongest of the three and the only one with no way back.
--
-- The check constraint is what makes *the rules are gone* structural rather than a rule
-- every writer has to remember: a deleted row cannot hold a spec, and a live row cannot
-- lack one. The same shape as `measurements.headline`, which exists only where the state
-- is current -- a reader cannot show what is not in the column.

alter table games alter column spec drop not null;
alter table games add column deleted timestamptz;
alter table games add constraint games_deleted_has_no_spec
    check ((deleted is null) <> (spec is null));

-- `name` is overwritten with the placeholder at the delete rather than translated by
-- each reader, so a match row, a badge, a milestone line and the profile's table all
-- read *deleted game* from the one query they already make. A name is part of what
-- somebody asked to have removed, and a convention that every reader must apply is a
-- convention one reader will forget.
