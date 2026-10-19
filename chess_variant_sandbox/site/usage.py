"""How the site is used, counted by the hour and never by the person.

`docs/plan-usage.md` §2. The handler times every dispatched request and hands it to
`Counter.record`, which adds it to one row per hour per route per outcome. Signed-in ids
are held in memory for the hour so the row can say how many different people made the
requests; the flush writes the number and drops the ids. No row names a person, holds an
address or reads a cookie, which is the privacy notice's promise and `migrations/0006`'s
shape.

`variant` is what a route template cannot say and the cost depends on. The two places
that know it call `note`, and the handler reads it back with `take` when the request ends.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from chess_variant_sandbox.site import quota, routes

KEEP_HOURS = timedelta(days=400)
"""How long `usage_hours` and `usage_people` rows are kept: a month against last year's."""

KEEP_SAMPLES = timedelta(days=90)
"""How long `usage_samples` rows are kept."""

SAMPLE_EVERY = 300.0
"""Seconds between two `usage_samples` rows, and between two flushes of the counter."""

MEMBER = "member"
VISITOR = "visitor"

# Anything a request did not match, including an invented path. A withheld route keeps
# its own template, so demand for it can be read.
OTHER = "other"

SESSION_ACTION = "POST /api/session/<id>/<action>"
SHOW_ME = ("POST /api/design/play", "show=rule")

# The exception classes that mean a limit refused the request.
REFUSALS = ("QuotaError", "Spent")

# The upper bound of each bucket but the last, which is everything over fifty.
BUCKET_TOPS = (1, 5, 10, 20, 50)

UTC = timezone.utc

_noted = threading.local()


def note(variant: str) -> None:
    """Name what this request's route template cannot, for the row it lands on."""
    _noted.variant = variant


def take() -> str:
    """The variant noted on this thread since the last `take`, and clear it."""
    variant = str(getattr(_noted, "variant", ""))
    _noted.variant = ""
    return variant


def template(method: str, path: str) -> str:
    """The route a request is, as method and template, so no hash or id enters a row."""
    found = routes.find(method, path)
    if found is None:
        parts = routes.split(path)
        found = next((one for one in routes.WITHHELD if one.matches(method, parts)), None)
    return OTHER if found is None else f"{found.method} {found.path}"


def default_variant(route: str, path: str) -> str:
    """The variant a request has before anything notes one: a session action's name."""
    if route == SESSION_ACTION:
        return routes.split(path)[-1]
    return ""


def hour_of(moment: float) -> datetime:
    """The UTC clock hour a Unix time falls in."""
    return datetime.fromtimestamp(moment, UTC).replace(minute=0, second=0, microsecond=0)


def buckets(counts: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    """How many people made 1, 2-5, 6-10, 11-20, 21-50 and more than 50 requests."""
    found = [0] * (len(BUCKET_TOPS) + 1)
    for count in counts:
        found[sum(1 for top in BUCKET_TOPS if count > top)] += 1
    one, two, six, eleven, twenty_one, over = found
    return one, two, six, eleven, twenty_one, over


@dataclass
class Sums:
    """One `usage_hours` row's measures."""

    count: int = 0
    cpu: float = 0.0
    wall: float = 0.0
    wall_max: float = 0.0

    def add(self, other: Sums) -> None:
        self.count += other.count
        self.cpu += other.cpu
        self.wall += other.wall
        self.wall_max = max(self.wall_max, other.wall_max)


@dataclass
class People:
    """The hour's signed-in ids for one route, in memory only."""

    made: dict[int, int] = field(default_factory=dict)
    refused: set[int] = field(default_factory=set)


HourKey = tuple[datetime, str, str, int, str, str]
PeopleKey = tuple[datetime, str, str]


@dataclass(frozen=True)
class PeopleRow:
    """One `usage_people` row, as numbers."""

    hour: datetime
    route: str
    variant: str
    people: int
    refused: int
    peak: int
    buckets: tuple[int, int, int, int, int, int]
    partial: bool


@dataclass(frozen=True)
class Drained:
    """What a flush writes: the finished hours' sums and their people rows."""

    hours: dict[HourKey, Sums]
    people: dict[PeopleKey, People]
    rows: tuple[PeopleRow, ...]


class Counter:
    """The site's usage in memory until the next flush, under one lock.

    `presses` is how the counter reaches the app's `quota.Presses`, which already holds each
    person's presses for the rolling hour, so `show=rule`'s people are read off it rather
    than off a second set. That budget also holds *show the stuck game* presses, so the
    `show=rule` row describes the shared allowance; `show=unfinished` counts its own people.
    """

    def __init__(
        self,
        presses: Callable[[], quota.Presses] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._presses = presses
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._hours: dict[HourKey, Sums] = {}
        self._people: dict[PeopleKey, People] = {}
        self._in_flight = 0
        self._in_flight_max = 0

    # ------------------------------------------------------------------ per request

    def record(
        self,
        route: str,
        variant: str,
        status: int,
        kind: str,
        viewer: int | None,
        cpu: float,
        wall: float,
    ) -> None:
        """Add one finished request to its hour's row."""
        hour = hour_of(self._clock())
        who = VISITOR if viewer is None else MEMBER
        with self._lock:
            key = (hour, route, variant, status, kind, who)
            self._hours.setdefault(key, Sums()).add(Sums(1, cpu, wall, wall))
            if viewer is None:
                return
            people = self._people.setdefault((hour, route, variant), People())
            people.made[viewer] = people.made.get(viewer, 0) + 1
            if kind in REFUSALS:
                people.refused.add(viewer)

    def enter(self) -> None:
        """A request has started."""
        with self._lock:
            self._in_flight += 1
            self._in_flight_max = max(self._in_flight_max, self._in_flight)

    def leave(self) -> None:
        """A request has finished."""
        with self._lock:
            self._in_flight -= 1

    def in_flight_max(self) -> int:
        """The most requests running at once since the last call."""
        with self._lock:
            peak = self._in_flight_max
            self._in_flight_max = self._in_flight
            return peak

    # ----------------------------------------------------------------------- flushing

    def drain(self, final: bool = False) -> Drained:
        """Take the finished hours out of memory, or every hour when the process is ending.

        An hour the process did not see whole is `partial`: it started inside it, or the
        flush comes before the hour ended.
        """
        now = self._clock()
        current = hour_of(now)
        with self._lock:
            hours = {k: v for k, v in self._hours.items() if final or k[0] < current}
            people = {k: v for k, v in self._people.items() if final or k[0] < current}
            for finished in hours:
                del self._hours[finished]
            for seen in people:
                del self._people[seen]
        pressed = self._presses().counts() if self._presses is not None else None
        rows = []
        for (hour, route, variant), found in sorted(people.items()):
            counts = list(found.made.values())
            if (route, variant) == SHOW_ME and pressed is not None:
                counts = pressed
            begun = hour.timestamp()
            partial = self._started > begun or now < begun + 3600
            rows.append(
                PeopleRow(
                    hour=hour,
                    route=route,
                    variant=variant,
                    people=len(counts),
                    refused=len(found.refused),
                    peak=max(counts, default=0),
                    buckets=buckets(counts),
                    partial=partial,
                )
            )
        return Drained(hours, people, tuple(rows))

    def restore(self, drained: Drained) -> None:
        """Put a drained batch back, when writing it failed."""
        with self._lock:
            for hour_key, sums in drained.hours.items():
                self._hours.setdefault(hour_key, Sums()).add(sums)
            for people_key, found in drained.people.items():
                into = self._people.setdefault(people_key, People())
                for user, made in found.made.items():
                    into.made[user] = into.made.get(user, 0) + made
                into.refused |= found.refused

    def flush(self, connection: Any, final: bool = False) -> Drained:
        """Write the finished hours and prune old rows. The caller commits.

        Upserts add to a row already there, so a restart inside an hour adds its half
        rather than replacing the first. Distinct counts cannot be added that way, so a
        row written twice is marked `partial`.
        """
        drained = self.drain(final)
        cutoff = datetime.fromtimestamp(self._clock(), UTC)
        try:
            with connection.cursor() as cursor:
                if drained.hours:
                    cursor.executemany(_HOURS_UPSERT, [
                        (*key, sums.count, sums.cpu, sums.wall, sums.wall_max)
                        for key, sums in sorted(drained.hours.items())
                    ])
                if drained.rows:
                    cursor.executemany(_PEOPLE_UPSERT, [
                        (row.hour, row.route, row.variant, row.people, row.refused,
                         row.peak, *row.buckets, row.partial)
                        for row in drained.rows
                    ])
                cursor.execute(
                    "delete from usage_hours where hour < %s", (cutoff - KEEP_HOURS,)
                )
                cursor.execute(
                    "delete from usage_people where hour < %s", (cutoff - KEEP_HOURS,)
                )
                cursor.execute(
                    "delete from usage_samples where at < %s", (cutoff - KEEP_SAMPLES,)
                )
        except BaseException:
            self.restore(drained)
            raise
        return drained


_HOURS_UPSERT = (
    "insert into usage_hours"
    " (hour, route, variant, status, kind, who, count, cpu, wall, wall_max)"
    " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    " on conflict (hour, route, variant, status, kind, who) do update set"
    " count = usage_hours.count + excluded.count,"
    " cpu = usage_hours.cpu + excluded.cpu,"
    " wall = usage_hours.wall + excluded.wall,"
    " wall_max = greatest(usage_hours.wall_max, excluded.wall_max)"
)

_PEOPLE_UPSERT = (
    "insert into usage_people"
    " (hour, route, variant, people, refused, peak, one, two_to_five, six_to_ten,"
    " eleven_to_twenty, twenty_one_to_fifty, over_fifty, partial)"
    " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    " on conflict (hour, route, variant) do update set"
    " people = usage_people.people + excluded.people,"
    " refused = usage_people.refused + excluded.refused,"
    " peak = greatest(usage_people.peak, excluded.peak),"
    " one = usage_people.one + excluded.one,"
    " two_to_five = usage_people.two_to_five + excluded.two_to_five,"
    " six_to_ten = usage_people.six_to_ten + excluded.six_to_ten,"
    " eleven_to_twenty = usage_people.eleven_to_twenty + excluded.eleven_to_twenty,"
    " twenty_one_to_fifty = usage_people.twenty_one_to_fifty"
    " + excluded.twenty_one_to_fifty,"
    " over_fifty = usage_people.over_fifty + excluded.over_fifty,"
    " partial = true"
)


class Ledger:
    """The process's CPU clock, shared out among the work running, by wall time (Q970).

    A request's own thread clock (`time.thread_time`) misses whatever it hands to other
    threads: the playability check plays its smoke games on the core's threads, and read
    0.48 of the unit's clock (`docs/reference/usage-first-month.md` §2). So a request is
    charged from the process's clock instead, which counts every thread the process
    runs. Between any two moments something starts or stops, the CPU the process spent
    is split evenly among what was running then, which is apportioning by wall time at
    the finest grain there is. One request alone -- nearly every request on the box,
    where `in_flight_max` was 0 or 1 in 98% of samples -- is charged exactly what the
    process spent while it ran.

    The process's clock and not the unit's: the unit's also counts child processes, and
    a spine rating (`site/spine.py`) runs in one for up to fifteen minutes; charging that
    to whoever made a move meanwhile would be a worse error than the one this fixes.

    Work beside a request that is not a request -- the legality runner's walks and spot
    checks -- takes a share of its own (`sharing`), so it absorbs its half rather than
    landing on the request. What still lands on a request: a share of anything else the
    process does while it runs (a restart's library warm, a page served outside `_json`),
    and, beside a second request, half of that one's work however unequal the two were.
    CPU spent with nothing running is charged to nobody: the idle floor.

    The share each participant has earned is one running sum, `_each`: a participant
    remembers it on entering, and its charge is the difference on leaving.
    """

    def __init__(self, clock: Callable[[], float] = time.process_time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._running = 0
        self._last = clock()
        self._each = 0.0

    def _advance(self) -> float:
        """Share out what the process spent since the last moment. Under the lock."""
        now = self._clock()
        if self._running:
            self._each += max(now - self._last, 0.0) / self._running
        self._last = now
        return self._each

    def enter(self) -> float:
        """Something has started; the mark its charge is measured from."""
        with self._lock:
            mark = self._advance()
            self._running += 1
            return mark

    def leave(self, mark: float) -> float:
        """Something has stopped; its CPU-seconds since `mark`."""
        with self._lock:
            charged = self._advance() - mark
            self._running -= 1
            return charged

    @contextmanager
    def sharing(self) -> Iterator[None]:
        """Run as a participant whose share is charged to nobody."""
        mark = self.enter()
        try:
            yield
        finally:
            self.leave(mark)


LEDGER = Ledger()
"""The process's one ledger, since the process has one clock."""


class Timing:
    """An endpoint call, timed and charged from `Ledger`, with what it raised.

    `thread` is the request's own thread clock, kept beside `cpu` so a test can tell the
    two apart; only `cpu` is written.
    """

    def __init__(self, produce: Callable[[], Any], ledger: Ledger | None = None) -> None:
        self._produce = produce
        self._ledger = LEDGER if ledger is None else ledger
        self.cpu = 0.0
        self.thread = 0.0
        self.wall = 0.0
        self.kind = ""
        self.variant = ""

    def __call__(self) -> Any:
        take()
        mark = self._ledger.enter()
        thread, wall = time.thread_time(), time.monotonic()
        try:
            return self._produce()
        except BaseException as error:
            self.kind = type(error).__name__
            raise
        finally:
            self.thread = time.thread_time() - thread
            self.wall = time.monotonic() - wall
            self.cpu = self._ledger.leave(mark)
            self.variant = take()


# ------------------------------------------------------------------------ the sampler


@dataclass(frozen=True)
class Reading:
    """What the machine's files said at one moment. `None` where a file is absent.

    `busy` and `idle` are `/proc/stat`'s cumulative jiffies and `app_usec` the unit's
    cumulative CPU, so a sample's figures are the difference of two readings.
    """

    busy: int | None = None
    idle: int | None = None
    app_usec: int | None = None
    load1: float | None = None
    mem_available: int | None = None
    app_memory: int | None = None
    disk_free: int | None = None


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii", errors="replace")
    except OSError:
        return None


def _field(text: str | None, name: str) -> int | None:
    """The integer after `name` on its own line, as `meminfo` and `cpu.stat` write them."""
    for line in (text or "").splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[0] == name and parts[1].isdigit():
            return int(parts[1])
    return None


def read_machine(root: Path = Path("/"), disk: Path = Path(".")) -> Reading:
    """Read `/proc`, this process's cgroup and the disk under `disk`, whatever exists.

    `root` stands in for `/` so a test can hand over a fake `/proc`. A file that is
    missing or unreadable gives `None` for its columns and never an error.
    """
    busy = idle = None
    stat = _text(root / "proc" / "stat")
    if stat:
        jiffies = [int(x) for x in stat.splitlines()[0].split()[1:] if x.isdigit()]
        if len(jiffies) >= 4:
            idle = jiffies[3] + (jiffies[4] if len(jiffies) > 4 else 0)
            busy = sum(jiffies[:8]) - idle

    load1 = None
    loadavg = _text(root / "proc" / "loadavg")
    if loadavg:
        try:
            load1 = float(loadavg.split()[0])
        except (ValueError, IndexError):
            load1 = None

    available = _field(_text(root / "proc" / "meminfo"), "MemAvailable")

    app_usec = app_memory = None
    group = _cgroup(_text(root / "proc" / "self" / "cgroup"))
    if group is not None:
        unit = root / "sys" / "fs" / "cgroup" / group
        app_usec = _field(_text(unit / "cpu.stat"), "usage_usec")
        current = (_text(unit / "memory.current") or "").strip()
        app_memory = int(current) // 2**20 if current.isdigit() else None

    try:
        disk_free: int | None = shutil.disk_usage(disk).free // 2**20
    except OSError:
        disk_free = None

    return Reading(
        busy=busy,
        idle=idle,
        app_usec=app_usec,
        load1=load1,
        mem_available=None if available is None else available // 1024,
        app_memory=app_memory,
        disk_free=disk_free,
    )


def _cgroup(text: str | None) -> str | None:
    """The unified-hierarchy path in `/proc/self/cgroup`, relative, or `None`."""
    for line in (text or "").splitlines():
        if line.startswith("0::"):
            return line[3:].strip().lstrip("/")
    return None


@dataclass(frozen=True)
class Sample:
    """One `usage_samples` row."""

    at: datetime
    cpu_busy: float | None
    app_cpu: float | None
    load1: float | None
    mem_available: int | None
    app_memory: int | None
    disk_free: int | None
    in_flight_max: int
    strong_seats_max: int | None


def between(before: Reading | None, after: Reading) -> tuple[float | None, float | None]:
    """The box's busy fraction and the unit's CPU-seconds from one reading to the next."""
    if before is None:
        return None, None
    cpu_busy = None
    if (
        before.busy is not None and before.idle is not None
        and after.busy is not None and after.idle is not None
    ):
        busy = after.busy - before.busy
        total = busy + after.idle - before.idle
        cpu_busy = busy / total if total > 0 else None
    app_cpu = None
    if before.app_usec is not None and after.app_usec is not None:
        app_cpu = (after.app_usec - before.app_usec) / 1e6
    return cpu_busy, app_cpu


Connect = Callable[[], AbstractContextManager[Any]]


class Sampler:
    """Every `SAMPLE_EVERY` seconds, one `usage_samples` row and a flush of the counter.

    A daemon thread started by `serve()`. `stop` wakes it, and it samples and flushes once
    more with every hour in memory, so a restart loses at most what arrived after that
    last flush. A failed tick prints one line and the thread carries on: usage counting
    never takes the site down.
    """

    def __init__(
        self,
        counter: Counter,
        connect: Connect,
        disk: Path = Path("."),
        root: Path = Path("/"),
        clock: Callable[[], float] = time.time,
        every: float = SAMPLE_EVERY,
    ) -> None:
        self.counter = counter
        self._connect = connect
        self._disk = disk
        self._root = root
        self._clock = clock
        self._every = every
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: Reading | None = None

    def sample(self, connection: Any) -> Sample:
        """Read the machine, write the row, and keep the reading for the next interval."""
        reading = read_machine(self._root, self._disk)
        cpu_busy, app_cpu = between(self._last, reading)
        self._last = reading
        row = Sample(
            at=datetime.fromtimestamp(self._clock(), UTC),
            cpu_busy=cpu_busy,
            app_cpu=app_cpu,
            load1=reading.load1,
            mem_available=reading.mem_available,
            app_memory=reading.app_memory,
            disk_free=reading.disk_free,
            in_flight_max=self.counter.in_flight_max(),
            strong_seats_max=quota.strong_seats(connection),
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "insert into usage_samples (at, cpu_busy, app_cpu, load1, mem_available,"
                " app_memory, disk_free, in_flight_max, strong_seats_max)"
                " values (%s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict (at) do nothing",
                (row.at, row.cpu_busy, row.app_cpu, row.load1, row.mem_available,
                 row.app_memory, row.disk_free, row.in_flight_max, row.strong_seats_max),
            )
        return row

    def tick(self, final: bool = False) -> None:
        """One sample and one flush, each in its own transaction, never raising.

        The flush goes second and alone, so a failed sample cannot roll back counts, and
        counts whose commit fails go back into memory for the next tick.
        """
        try:
            with self._connect() as connection:
                self.sample(connection)
        except Exception as error:  # noqa: BLE001 - a sample must not kill the server
            print(f"usage: the sample was not written: {error}")
        drained: Drained | None = None
        try:
            with self._connect() as connection:
                drained = self.counter.flush(connection, final)
        except Exception as error:  # noqa: BLE001 - nor must a flush
            if drained is not None:
                self.counter.restore(drained)
            print(f"usage: the counts were not written: {error}")

    def start_reading(self) -> None:
        """Take the reading the first sample's CPU figures are measured from."""
        self._last = read_machine(self._root, self._disk)

    def start(self) -> threading.Thread:
        """Take the first reading and start the thread."""
        self.start_reading()
        self._thread = threading.Thread(target=self._run, name="usage", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, timeout: float = 30.0) -> None:
        """Wake the thread for its last sample and flush, and wait for them."""
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stopping.wait(self._every):
            self.tick()
        self.tick(final=True)


# ------------------------------------------------------------------------ the readings
#
# `docs/plan-usage.md` §4.3. The admin page and `chess-variant-sandbox site usage` both read through
# `report` and `samples`, and both print the same sentences, so one question has one
# number. Rows are fetched plainly and summed here, which keeps the arithmetic testable
# without a database.

GIGABYTE = 1024
"""Memory available under this many MB is a sample §1.4 counts."""

PRESS_LINES = (10, 20, 50)
"""The person-hour thresholds §1.1 reads the *show me* buckets against."""

ROUTE_CANNOT = (
    "People are signed-in people. A visitor is counted as requests and never as a person.",
    "People-hours count a person once for every hour they were active, so they are not "
    "how many different people came.",
    "CPU is the process's clock shared by wall time among the requests running at once, "
    "so a route's work on other threads is charged to it. Rows written before that "
    "change (Q970, 2026-10-19) timed the request's own thread and missed that work.",
)

SHOW_ME_CANNOT = (
    "The buckets are clock hours and the limit counts a rolling hour, so a person who "
    "pressed across two clock hours is split between them.",
    "The people, buckets and peak cover show me and show the stuck game together, "
    "because the two buttons share one allowance; the press count is show me alone.",
    "CPU per press is a mean. The slowest press is known in wall seconds only.",
)

PARTIAL_CANNOT = (
    "A partial hour is one the process did not see whole. Its people are the halves "
    "either side of a restart added together, so a person may count twice, and it is "
    "never added to the whole hours.",
)

BOX_CANNOT = (
    "Strong seats are the seats live at the moment of each sample, not the most at "
    "once in the five minutes between.",
    "The machine's columns are empty where there is no /proc, as on a development "
    "machine.",
)


@dataclass(frozen=True)
class RouteUse:
    """One route and variant over a period: what was asked of the site, and by how many.

    `people_*` are read from whole hours only. A partial hour's people are kept apart in
    `partial_people`, so a restart's double count never reaches the whole-hour figures.
    """

    route: str
    variant: str
    requests: int
    refused: int
    by_visitors: int
    cpu: float
    wall: float
    wall_max: float
    people_largest: int
    people_hours: int
    partial_hours: int
    partial_people: int

    @property
    def wall_mean(self) -> float:
        return self.wall / self.requests if self.requests else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "variant": self.variant,
            "requests": self.requests,
            "refused": self.refused,
            "byVisitors": self.by_visitors,
            "cpu": round(self.cpu, 3),
            "wallMean": round(self.wall_mean, 4),
            "wallMax": round(self.wall_max, 4),
            "peopleLargest": self.people_largest,
            "peopleHours": self.people_hours,
            "partialHours": self.partial_hours,
            "partialPeople": self.partial_people,
        }


@dataclass(frozen=True)
class ShowMe:
    """§1.1: how hard people press against the *show me* allowance."""

    presses: int
    spent: int
    asked_by_visitors: int
    cpu: float
    buckets: tuple[int, int, int, int, int, int]
    hours: int
    peak: int
    partial_hours: int
    partial_buckets: tuple[int, int, int, int, int, int]

    def over(self, presses: int) -> int:
        """Person-hours with more than `presses` presses, from whole hours only."""
        return sum(
            n for n, top in zip(self.buckets, (*BUCKET_TOPS, None), strict=True)
            if top is None or top > presses
        )

    def said(self) -> list[str]:
        """The reading as sentences, the same on the page and in the command."""
        per_press = f"{self.cpu / self.presses:.2f}" if self.presses else "no"
        one, two, six, eleven, twenty_one, over_fifty = self.buckets
        lines = [
            f"{self.presses} show me presses served, {per_press} CPU-seconds each on "
            f"average; {self.spent} refused past the hourly limit; "
            f"{self.asked_by_visitors} asked for by visitors, who must sign in.",
            f"Over {self.hours} whole hour(s), person-hours by presses: 1: {one}, "
            f"2-5: {two}, 6-10: {six}, 11-20: {eleven}, 21-50: {twenty_one}, "
            f"over 50: {over_fifty}.",
            "Person-hours over "
            + ", ".join(f"{n}: {self.over(n)}" for n in PRESS_LINES)
            + f". The most one person pressed in a rolling hour: {self.peak}.",
        ]
        if self.partial_hours:
            lines.append(
                f"{self.partial_hours} partial hour(s) are left out of the above; their "
                f"buckets, halves added, are {', '.join(map(str, self.partial_buckets))}."
            )
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "presses": self.presses,
            "spent": self.spent,
            "askedByVisitors": self.asked_by_visitors,
            "cpu": round(self.cpu, 3),
            "buckets": list(self.buckets),
            "over": {str(n): self.over(n) for n in PRESS_LINES},
            "hours": self.hours,
            "peak": self.peak,
            "partialHours": self.partial_hours,
            "partialBuckets": list(self.partial_buckets),
            "said": self.said(),
        }


@dataclass(frozen=True)
class Report:
    """§1.1 and §1.2 over a period: requests, refusals, CPU and people by route."""

    since: datetime
    until: datetime
    hours: int
    partial: tuple[datetime, ...]
    routes: tuple[RouteUse, ...]
    show_me: ShowMe

    def said(self) -> list[str]:
        head = (
            f"From {_minute(self.since)} to {_minute(self.until)} UTC: {self.hours} hour(s) "
            f"with requests, {len(self.partial)} of them partial."
        )
        return [head]

    def cannot(self) -> list[str]:
        return [*ROUTE_CANNOT, *SHOW_ME_CANNOT, *PARTIAL_CANNOT]

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "hours": self.hours,
            "partial": [hour.isoformat() for hour in self.partial],
            "said": self.said(),
            "routes": [one.as_dict() for one in self.routes],
            "showMe": self.show_me.as_dict(),
            "cannot": self.cannot(),
        }


@dataclass(frozen=True)
class Samples:
    """§1.4 over a period: what the machine had left, every five minutes."""

    since: datetime
    until: datetime
    rows: tuple[Sample, ...]

    @property
    def newest(self) -> Sample | None:
        return self.rows[-1] if self.rows else None

    @property
    def busiest(self) -> Sample | None:
        """The five minutes with the box busiest."""
        found = [row for row in self.rows if row.cpu_busy is not None]
        return max(found, key=lambda row: row.cpu_busy or 0.0) if found else None

    @property
    def leanest(self) -> Sample | None:
        """The five minutes with the least memory available."""
        found = [row for row in self.rows if row.mem_available is not None]
        return min(found, key=lambda row: row.mem_available or 0) if found else None

    @property
    def under_a_gigabyte(self) -> int:
        return sum(
            1 for row in self.rows
            if row.mem_available is not None and row.mem_available < GIGABYTE
        )

    @property
    def strong_seats_seen(self) -> int | None:
        """The most strong seats any one sample found live. Samples, not the time between."""
        found = [row.strong_seats_max for row in self.rows if row.strong_seats_max is not None]
        return max(found) if found else None

    @property
    def in_flight_max(self) -> int | None:
        found = [row.in_flight_max for row in self.rows]
        return max(found) if found else None

    def said(self) -> list[str]:
        if not self.rows:
            return [
                f"No samples from {_minute(self.since)} to {_minute(self.until)} UTC."
            ]
        newest, busiest, leanest = self.newest, self.busiest, self.leanest
        assert newest is not None
        lines = [
            f"{len(self.rows)} sample(s) from {_minute(self.since)} to "
            f"{_minute(self.until)} UTC. The newest, at {_minute(newest.at)}: "
            f"{_sample_text(newest)}.",
            "The busiest five minutes: "
            + (f"{_minute(busiest.at)}, {_sample_text(busiest)}." if busiest else "unknown."),
            "The least memory left: "
            + (f"{_minute(leanest.at)}, {_sample_text(leanest)}." if leanest else "unknown."),
            f"Memory available was under 1 GB in {self.under_a_gigabyte} sample(s). "
            f"The most requests running at once: {self.in_flight_max}. "
            f"The most strong seats live at a sample: {_or_unknown(self.strong_seats_seen)}.",
        ]
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since.isoformat(),
            "until": self.until.isoformat(),
            "count": len(self.rows),
            "newest": sample_dict(self.newest),
            "busiest": sample_dict(self.busiest),
            "leanest": sample_dict(self.leanest),
            "underAGigabyte": self.under_a_gigabyte,
            "inFlightMax": self.in_flight_max,
            "strongSeatsAtASample": self.strong_seats_seen,
            "said": self.said(),
            "cannot": list(BOX_CANNOT),
        }


def period(days: float, now: float | None = None) -> tuple[datetime, datetime]:
    """The `days` before the start of the current hour, so every hour in it has ended."""
    until = hour_of(time.time() if now is None else now)
    return until - timedelta(days=days), until


def report(
    connection: Any, since: datetime, until: datetime, route: str | None = None
) -> Report:
    """Requests, refusals, CPU, wall time and people by route, over `[since, until)`."""
    where = " where hour >= %s and hour < %s"
    params: tuple[Any, ...] = (since, until)
    if route is not None:
        where += " and route = %s"
        params = (*params, route)
    with connection.cursor() as cursor:
        cursor.execute(
            "select hour, route, variant, status, kind, who, count, cpu, wall, wall_max"
            " from usage_hours" + where + " order by hour",
            params,
        )
        hours = list(cursor.fetchall())
        cursor.execute(
            "select hour, route, variant, people, refused, peak, one, two_to_five,"
            " six_to_ten, eleven_to_twenty, twenty_one_to_fifty, over_fifty, partial"
            " from usage_people" + where + " order by hour",
            params,
        )
        people = list(cursor.fetchall())
    return summarise(since, until, hours, people)


def summarise(
    since: datetime, until: datetime, hours: Sequence[Any], people: Sequence[Any]
) -> Report:
    """Fold `usage_hours` and `usage_people` rows, as the two selects return them."""
    sums: dict[tuple[str, str], list[Any]] = {}
    seen: set[datetime] = set()
    for hour, route, variant, _status, kind, who, count, cpu, wall, wall_max in hours:
        seen.add(hour)
        into = sums.setdefault((route, variant), [0, 0, 0, 0.0, 0.0, 0.0, set()])
        into[0] += int(count)
        into[1] += int(count) if kind in REFUSALS else 0
        into[2] += int(count) if who == VISITOR else 0
        into[3] += float(cpu)
        into[4] += float(wall)
        into[5] = max(into[5], float(wall_max))
        into[6].add(hour)

    heads: dict[tuple[str, str], list[int]] = {}
    partial: set[datetime] = set()
    show_buckets = [0] * 6
    show_partial = [0] * 6
    show_hours = show_partial_hours = show_peak = 0
    for hour, route, variant, count, _refused, peak, *six, is_partial in people:
        into_people = heads.setdefault((route, variant), [0, 0, 0, 0])
        if is_partial:
            partial.add(hour)
            into_people[2] += 1
            into_people[3] += int(count)
        else:
            into_people[0] = max(into_people[0], int(count))
            into_people[1] += int(count)
        if (route, variant) == SHOW_ME:
            show_peak = max(show_peak, int(peak))
            target = show_partial if is_partial else show_buckets
            for i, n in enumerate(six):
                target[i] += int(n)
            if is_partial:
                show_partial_hours += 1
            else:
                show_hours += 1

    uses = []
    for key in sorted(set(sums) | set(heads)):
        requests, refused, visitors, cpu, wall, wall_max, _ = sums.get(
            key, [0, 0, 0, 0.0, 0.0, 0.0, set()]
        )
        largest, people_hours, partial_hours, partial_people = heads.get(key, [0, 0, 0, 0])
        uses.append(RouteUse(
            route=key[0], variant=key[1], requests=requests, refused=refused,
            by_visitors=visitors, cpu=cpu, wall=wall, wall_max=wall_max,
            people_largest=largest, people_hours=people_hours,
            partial_hours=partial_hours, partial_people=partial_people,
        ))
    uses.sort(key=lambda one: (-one.cpu, -one.requests, one.route, one.variant))

    def show_count(status: int, kind: str, who: str | None) -> int:
        return sum(
            int(row[6]) for row in hours
            if (row[1], row[2]) == SHOW_ME and row[3] == status and row[4] == kind
            and (who is None or row[5] == who)
        )

    show_cpu = sum(float(row[7]) for row in hours if (row[1], row[2]) == SHOW_ME)
    show_me = ShowMe(
        presses=show_count(200, "", None),
        spent=show_count(429, "Spent", None),
        asked_by_visitors=show_count(400, "QuotaError", VISITOR),
        cpu=show_cpu,
        buckets=_six(show_buckets),
        hours=show_hours,
        peak=show_peak,
        partial_hours=show_partial_hours,
        partial_buckets=_six(show_partial),
    )
    return Report(
        since=since,
        until=until,
        hours=len(seen | partial),
        partial=tuple(sorted(partial)),
        routes=tuple(uses),
        show_me=show_me,
    )


def samples(connection: Any, since: datetime, until: datetime) -> Samples:
    """Every `usage_samples` row in `[since, until)`, oldest first."""
    with connection.cursor() as cursor:
        cursor.execute(
            "select at, cpu_busy, app_cpu, load1, mem_available, app_memory, disk_free,"
            " in_flight_max, strong_seats_max from usage_samples"
            " where at >= %s and at < %s order by at",
            (since, until),
        )
        rows = cursor.fetchall()
    return Samples(since, until, tuple(
        Sample(
            at=row[0],
            cpu_busy=None if row[1] is None else float(row[1]),
            app_cpu=None if row[2] is None else float(row[2]),
            load1=None if row[3] is None else float(row[3]),
            mem_available=row[4],
            app_memory=row[5],
            disk_free=row[6],
            in_flight_max=0 if row[7] is None else int(row[7]),
            strong_seats_max=row[8],
        )
        for row in rows
    ))


def admin_view(connection: Any, now: float | None = None) -> dict[str, Any]:
    """The `usage` key on `GET /api/admin`: the last day, the last month and the box."""
    moment = time.time() if now is None else now
    day, month = period(1, moment), period(30, moment)
    at = datetime.fromtimestamp(moment, UTC)
    return {
        "day": report(connection, *day).as_dict(),
        "month": report(connection, *month).as_dict(),
        "box": samples(connection, at - timedelta(days=7), at).as_dict(),
        "kept": (
            f"Hourly counts are kept {KEEP_HOURS.days} days and samples "
            f"{KEEP_SAMPLES.days} days."
        ),
    }


def render(found: Report, box: Samples, listed: bool = False) -> str:
    """The readings as text, for `chess-variant-sandbox site usage`."""
    lines = [*found.said(), ""]
    header = (
        f"{'route':<46} {'requests':>8} {'refused':>7} {'visitor':>7} {'cpu s':>9} "
        f"{'mean s':>7} {'max s':>7} {'people':>6} {'p-hours':>7} {'partial':>7}"
    )
    lines.append(header)
    for one in found.routes:
        name = f"{one.route} {one.variant}".strip()
        partial = f"{one.partial_hours}h/{one.partial_people}" if one.partial_hours else "-"
        lines.append(
            f"{name:<46} {one.requests:>8} {one.refused:>7} {one.by_visitors:>7} "
            f"{one.cpu:>9.2f} {one.wall_mean:>7.3f} {one.wall_max:>7.3f} "
            f"{one.people_largest:>6} {one.people_hours:>7} {partial:>7}"
        )
    if not found.routes:
        lines.append("(no requests counted)")
    lines += ["", "Show me (plan-usage 1.1):", *(f"  {line}" for line in found.show_me.said())]
    lines += ["", "The box (plan-usage 1.4):", *(f"  {line}" for line in box.said())]
    if listed:
        lines.append("")
        for row in box.rows:
            lines.append(f"  {_minute(row.at)}  {_sample_text(row)}")
    lines += ["", "What these numbers cannot say:"]
    lines += [f"  - {line}" for line in (*found.cannot(), *BOX_CANNOT)]
    return "\n".join(lines)


def _six(values: Sequence[int]) -> tuple[int, int, int, int, int, int]:
    one, two, six, eleven, twenty_one, over = values
    return one, two, six, eleven, twenty_one, over


def _minute(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M")


def _or_unknown(value: object) -> str:
    return "unknown" if value is None else str(value)


def _sample_text(row: Sample) -> str:
    busy = "unknown" if row.cpu_busy is None else f"{row.cpu_busy:.0%}"
    app_cpu = "unknown" if row.app_cpu is None else f"{row.app_cpu:.1f}"
    return (
        f"box busy {busy}, app CPU {app_cpu} s, "
        f"memory left {_or_unknown(row.mem_available)} MB, "
        f"app memory {_or_unknown(row.app_memory)} MB, "
        f"disk free {_or_unknown(row.disk_free)} MB, "
        f"strong seats live {_or_unknown(row.strong_seats_max)}"
    )


def sample_dict(row: Sample | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "at": row.at.isoformat(),
        "cpuBusy": row.cpu_busy,
        "appCpu": row.app_cpu,
        "load1": row.load1,
        "memAvailable": row.mem_available,
        "appMemory": row.app_memory,
        "diskFree": row.disk_free,
        "inFlightMax": row.in_flight_max,
        "strongSeatsAtSample": row.strong_seats_max,
    }
