-- Tier two, the spot check, and what it needs to hold (Q503e, `docs/plan-client-compute.md`
-- tier two, `docs/plan-game-pipeline.md` Q503e, `site/spotcheck.py`).
--
-- **`jobs.result`** is the provisional result a `sample` job replays. Tier one's job makes
-- the result; the sample job that follows it names it here, because `provisional.job` is
-- the queue row that last *moved* the result and a queued check has not moved it yet. It
-- cascades: a result that goes takes its unfinished check with it.
--
-- **`jobs.sample`** is the draw: an HMAC of the submission's text under a key the server
-- drew once and never kept, as hex. Which games are replayed is a function of it alone
-- (`spotcheck.draw`), so the sample is the same when drawn twice from the same text and
-- differs when one byte of the text differs, and nobody who sent the text can know it in
-- advance. Null on every other kind of job.
--
-- **`jobs.cpu_seconds`** is the processor time the job's own thread spent, tier one's walk
-- and tier two's replay alike, read with `time.thread_time`. A person's day is the sum of
-- theirs (`quota.CHECK_CPU_A_DAY`), priced in CPU-seconds because one check of the
-- library's largest board costs what ten of chess do (`docs/reference/tier-costs-on-the-box.md`).
--
-- **`users.flagged`** is when a spot check found a submitted game the server's replay did
-- not reproduce (`provisional.mismatched`). A flagged account cannot publish, and what it
-- sends later is walked by tier one but never spot-checked, so it waits for the server's
-- own measurement, until an admin clears the stamp. Null for everybody here.
--
-- **`provisional.stale_since`** is when the server first found a standing result's stamp
-- no longer matches its basis (`provisional.mark_stale`, at every start of the runner and
-- on every read of the shortlist bar). The pipeline's §2.4 gives a trial game whose reading
-- is stale 30 days before it stops counting toward the shortlist (`quota.STALE_GRACE_DAYS`);
-- this is the clock. Cleared if the basis comes back (a rolled-back deploy).

alter table jobs add column result bigint references provisional (id) on delete cascade;
alter table jobs add column sample text check (sample is null or sample ~ '^[0-9a-f]{64}$');
alter table jobs add column cpu_seconds double precision check (cpu_seconds >= 0);
alter table users add column flagged timestamptz;
alter table provisional add column stale_since timestamptz;

-- A person's checking today, which every filing asks.
create index jobs_by_requester on jobs (requester, created);
