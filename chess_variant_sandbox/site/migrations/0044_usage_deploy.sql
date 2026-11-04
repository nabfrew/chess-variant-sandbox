-- The deploy's own requests get a `who` of their own (Q1091).
--
-- `deploy/remote/deploy.sh`'s `restart` step warms the library, the map and the games list
-- on the box, unsigned, and those reads were counted as a visitor's: nearly all of the
-- visitor rows' library CPU in the first month was the deploy's cold read
-- (`docs/reference/usage-first-month.md` §4.3). `site/usage.py`'s `from_deploy` now tells
-- them apart, and their rows are written with `who = 'deploy'`. Rows before this are left
-- as they were: a reading of them still takes the restart hours out.

alter table usage_hours drop constraint usage_hours_who_check;
alter table usage_hours add constraint usage_hours_who_check
    check (who in ('member', 'visitor', 'deploy'));
