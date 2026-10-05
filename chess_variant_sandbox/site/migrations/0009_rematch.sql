-- Which match this one was a rematch of, and null everywhere else.
--
-- Box C1 of `docs/plan-meta-game.md`. Phase 4's third prediction is that the games people
-- play again correlate with `drama` rather than with the headline, and no row said a
-- match was a rematch. One nullable reference makes that readable.
--
-- The column is written by `matches.start` when the board's rematch control starts the
-- next game from a board that is already on screen, and by nothing else. A match started
-- from a game page, from the designer or from the command line has it null. So *null*
-- means *not a rematch*, not *unknown*, and a count over the column is a real number
-- rather than a floor.
--
-- The row it names is the person's own previous match on the same game -- `start` checks
-- both before writing it -- so the column carries no fact about a player the two rows do
-- not already hold. A rematch is a fresh match in every other respect: its own id, seats,
-- seed, moves and rating update, and the older row is untouched.
--
-- No delete action, because nothing deletes a match row. Leaving empties a person out of
-- their rows and the sweep resigns rather than deletes (`0005_leaving.sql`), since every
-- rating table is a fold over these rows from the first. A game that is deleted takes its
-- matches with it through `matches.game`, and both ends of a chain are on the same game,
-- so the two sides always go together.

alter table matches add column rematch_of text references matches (id);

-- The reading is per match and by primary key -- a win asks what it followed -- so the
-- index the column needs is the one that exists. This partial index is for the other
-- direction, counting the rematches on a game, which is the prediction's own question.

create index matches_rematched on matches (game) where rematch_of is not null;
