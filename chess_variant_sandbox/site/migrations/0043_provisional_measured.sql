-- A visitor's one run can now read *not measured* (RS3, `docs/plan-teacher.md` §10.16.4
-- point 4).
--
-- A run the roster could not read, and that failed no gate it could read, reaches no
-- verdict either way: it is not ruled out, and it is not admitted. `viable` stays false for
-- it, as it was not admitted, and this column says which of the two a false `viable` is.
-- Every row written before it was measured in the old sense, so the default is true.

alter table provisional add column measured boolean not null default true;
