-- A library game whose spec has left the tree, kept rather than deleted.
--
-- A spec is retired from the corpus when something supersedes it, and the deployed tree
-- is then the only account of what the library holds. The row cannot go: measurements,
-- ratings, votes and match rows reference it, and a match already played is a fact about
-- somebody's afternoon. So the loader stamps `retired` instead, and clears it again if
-- the spec comes back under the same mechanics hash.
--
-- `retired` is weaker than `revoked` and is a separate column because the two are
-- separate acts. `revoked` is 1.8's kill switch, set by an admin, and it takes the game
-- off its own page and out of everything. `retired` is the loader's reading of the tree:
-- off the listing and refused for a new match, while the row, its matches and its
-- ratings stay readable. An admin's switch and a deploy's reading must not overwrite
-- each other, which one shared column could not avoid.
--
-- Only `corpus` and `archive` rows are ever stamped here. A draft saved on the site has
-- no spec in the tree by definition, and retiring one would take down every draft on the
-- first deploy after it was saved.

alter table games add column retired timestamptz;
