-- Dismiss a report (Q503o, follows `0017_reports.sql`).
--
-- A report the maintainer has read and judged false or handled would sit on `/admin` for
-- good, and deleting its row would free the same person to report the same game again.
-- `dismissed` is the answer: the row stays, so (reporter, game) is still taken and
-- `reports.of` still finds it, and `reports.every` leaves it off the list.
-- Null is an open report; a timestamp is when it was dismissed.

alter table reports add column dismissed timestamptz;
