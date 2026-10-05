-- Which net a learned seat really played (Q709, `docs/reference/intransitive-quiet-endgame.md`
-- §b.5).
--
-- A learned player's agent string names a *training procedure*, the same on every board,
-- so it does not say which net sat down, and before Q709 a learned seat whose net could
-- not be loaded played the built-in evaluation under the learned player's name. Now such a
-- seat refuses to start or to move (`nets.require_net`), and **`matches.seat0_played` and
-- `seat1_played`** record what did play: `net:<12 hex>`, the SHA-256 of the net's bytes
-- (`NetRecord.digest`, as a measurement's `Basis.net` stamps it), written when the match
-- starts. Null for every other seat, a person's or a computer player whose agent string
-- already says everything about it, and for every row written before this file.
--
-- Each reply checks the net it is about to play against the column and refuses on a
-- mismatch, so a row can never say *learned* beside a move the built-in evaluation chose.

alter table matches add column seat0_played text;
alter table matches add column seat1_played text;
