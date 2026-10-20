-- The `links` table retires: near has one meaning, the map's.
--
-- `0012_links.sql` kept every pair of the store's games with `spec/relate.py`'s count of
-- rules apart, and `0013_similarity.sql` added its similarity index, because a game's page
-- ranked its neighbours by them and the store was where that list could be narrowed to
-- what each person may see. M5 of `docs/plan-map.md` gives *near* one meaning everywhere
-- the site names one game as near or like another: the map's likeness
-- (`spec/likeness.py`), over the map's own games (`web/graphing.near`), which are the
-- same for everybody. No page ranks by this table now, the load and a draft's save no
-- longer write it, and every row in it was derived from the rules the `games` table
-- still holds, so nothing is lost with it.

drop table if exists links;
