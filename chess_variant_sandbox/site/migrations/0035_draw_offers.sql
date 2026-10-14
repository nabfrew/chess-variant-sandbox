-- Draw offers, logged (Q790, `docs/reference/intransitive-quiet-endgame.md` §e.1).
--
-- A person may offer a draw in every game: to the computer, which answers at once by the
-- strongest search the game has (`web/draw_offer.py`, through the core), or to the other
-- person, as Q570 already allowed. **`draw_offers`** is every offer made on this row, in
-- order, one object each: `seat` and `ply` always; against the computer also `accepted`,
-- and the core's own `value`, `depth` and the `band` the rule read them against, so a
-- verdict can be checked later against what the core said rather than worked out again.
--
-- It is what the one-offer-in-ten-plies rule reads, which is why it is a log rather than
-- one column: a declined offer leaves no standing offer behind, and the next one has to
-- know when it was made. Null on a row nobody has offered a draw in, and on every row
-- written before this; a restart empties it, since the game starts again.
--
-- A draw both sides agreed -- accepted by the computer or by the other person -- is
-- finished `result = 0.5`, `reason = 'agreed'` (`0022_correspondence.sql`), and rated as a
-- draw wherever the row is rated.

alter table matches add column draw_offers jsonb;
