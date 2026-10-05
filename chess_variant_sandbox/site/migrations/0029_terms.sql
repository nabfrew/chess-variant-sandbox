-- What a designer agreed to when they published a game (Q722, `site/terms.py`).
--
-- **`games.terms_accepted`** is when the designer ticked the box agreeing to the terms and
-- pressed *Publish*, and **`games.terms_version`** is which version of the terms
-- (`terms.VERSION`) that box stood for. Both or neither: a stamp with no version would
-- not say what was agreed. A tick and a number, no free text.
--
-- Rows published before the terms existed are left null, and nothing marks them later:
-- nobody agreed to anything for them, and the record says so. A resting game brought back
-- is a new publish and is stamped again with the version agreed then.

alter table games add column terms_accepted timestamptz;
alter table games add column terms_version integer check (terms_version >= 1);
alter table games add constraint games_terms_both
    check ((terms_accepted is null) = (terms_version is null));
