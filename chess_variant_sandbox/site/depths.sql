-- The site's queues: how many wait in each, how long the oldest has waited, and whether
-- that reads as a problem (Q813). One statement, read-only, with no parameters, and the
-- only place a queue or its threshold is written down. Two readers run it unchanged:
-- `site/depths.py` for `/admin`'s *Status*, and `deploy/status.sh`, which feeds this file
-- to the box's `psql`. So the page and the script print the same numbers by construction.
--
-- Rules for editing it, because both readers depend on them:
--   * no `%` and no psql `:name` (psycopg runs it with no parameters, psql reads it raw);
--   * no `|` and no line break inside a label or a reason (`psql -A` separates with `|`);
--   * the column order below is the contract: `status.sh` reads the first three.
--
-- A threshold is on the oldest's age, never on the depth alone: a queue with one runner
-- that is moving is only slower when it is deep, and what a waiting person feels is how
-- long. The age is from the row's own moment (filed, pressed, asked, reported) except the
-- sweep's, which is how long past its limit a match has gone unresigned. `now()` is the
-- database's clock, which every query in the site measures against.
--
-- Not here, on purpose: the refresher's owed repairs (it runs on the development machine,
-- not the box, and `/admin`'s *The refresher* already reads its status file against its
-- own quiet limit); move emails and invitations (sent at once, never queued); suggestions
-- (nobody waits on an answer); the pipeline worker's heartbeat (no worker exists, Q503g).

with limits (place, queue, label, limit_seconds, limit_said, why) as (values
    (1, 'checks', 'checks of submitted scores', 1800, '30 min',
     'One runner takes them in turn, ahead of all other work. A walk costs about ten '
     'CPU-seconds and a spot check is stopped at ten minutes (quota.SAMPLE_JOB_WALL), so '
     'a check waiting three spot checks long means the runner has stopped or is flooded.'),
    (2, 'publishes', 'publishes waiting for their check', 3600, '1 h',
     'A publish waits only for its own game''s walk and spot check, which run first. '
     'Twice the checks'' limit means a press nothing is going to settle.'),
    (3, 'ratings', 'computer players rated at publish', 7200, '2 h',
     'One game at a time after the checks, each stopped at fifteen minutes '
     '(quota.SPINE_WALL). Eight ratings long is a stuck runner or a backlog to look at.'),
    (4, 'sweep', 'matches past their idle limit', 1800, '30 min',
     'A timer resigns them every ten minutes. Three ticks overdue means that timer has '
     'stopped.'),
    (5, 'asks', 'requests for an invitation', 604800, '7 days',
     'A person waiting for an answer, told nothing meanwhile. A week unanswered reads '
     'as ignored.'),
    (6, 'reports', 'open reports', 172800, '2 days',
     'A reported game stays listed until somebody looks. Two days is the longest a game '
     'reported as spam or offensive should stay up unread.'),
    (7, 'training', 'games queued for training', null, 'none',
     'Nothing takes these until the pipeline worker exists (Q503g), so a depth here is '
     'games waiting for that worker, not a fault.')
),
found (queue, depth, since) as (
    select 'checks', count(*), min(created) from jobs
        where kind in ('legality', 'sample') and state in ('queued', 'running')
    union all
    select 'publishes', count(*), min(publish_asked) from games
        where publish_asked is not null
    union all
    select 'ratings', count(*), min(created) from jobs
        where kind = 'spine' and state in ('queued', 'running')
    union all
    -- `matches.sweep`'s two rules: a game between two people goes to the player to move
    -- after five days (quota.CORRESPONDENCE_IDLE); a leaderboard match with a computer in
    -- exactly one seat is resigned after thirty (quota.IDLE_LIMIT). `since` is when it
    -- became due.
    select 'sweep', count(*), min(coalesce(moved, started) + case
            when seat0_user is not null and seat1_user is not null then interval '5 days'
            else interval '30 days' end)
        from matches
        where finished is null and source is null and (
            (seat0_user is not null and seat1_user is not null
             and coalesce(moved, started) < now() - interval '5 days')
            or (mode = 'leaderboard'
                and (seat0_agent is not null) <> (seat1_agent is not null)
                and coalesce(moved, started) < now() - interval '30 days'))
    union all
    select 'asks', count(*), min(created) from invite_asks where answered is null
    union all
    select 'reports', count(*), min(created) from reports where dismissed is null
    union all
    select 'training', count(*), min(created) from jobs
        where kind = 'train' and state in ('queued', 'running')
),
aged as (
    select queue, depth,
        case when since is not null
            then greatest(0, floor(extract(epoch from now() - since)))::bigint
        end as age
    from found
),
worded as (
    select l.place, l.queue, l.label, a.depth, a.age, l.limit_seconds, l.limit_said, l.why,
        coalesce(a.age > l.limit_seconds, false) as problem,
        case
            when a.age is null then null
            when a.age < 120 then a.age || ' s'
            when a.age < 7200 then (a.age / 60) || ' min'
            when a.age < 172800 then (a.age / 3600) || ' h'
            else (a.age / 86400) || ' days'
        end as oldest_said
    from limits l join aged a using (queue)
)
select queue, problem,
    label || ': ' || case
        when depth = 0 then 'none'
        else depth || ', the oldest ' || oldest_said
    end || case
        when limit_seconds is null then ' (no limit yet)'
        when problem then ' -- PROBLEM, past ' || limit_said
        else ' (a problem past ' || limit_said || ')'
    end as said,
    label, depth, age, oldest_said, limit_seconds, limit_said, why
from worded
order by place;
