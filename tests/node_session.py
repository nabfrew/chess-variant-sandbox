"""One `node` for the whole session, running each page check in a fresh `vm` context.

The page, the designer and the viz each ship JavaScript that nothing type-checks, so about
two dozen helpers across the suite run it in node against a stubbed DOM and read what it
printed. Each used to write a file and start `node` on it: 377 starts in the fast suite, at
about 0.13 s a start, 49 s of the 384 s Q377 measured. Most checks re-run the same 580 KB
page script.

`run(source)` hands the source to one long-lived node instead, which runs it the way
`node file.js` would -- as a CommonJS module body in a context of its own, with node's
globals, a `console` whose output is collected, and timers that are tracked so the job ends
when nothing it scheduled is left, which is when the process would have exited. What comes
back has the three fields the helpers read off `subprocess.run`: `returncode` (0, or 1 for
an uncaught error, a rejection nobody handled, or a syntax error), `stdout` and `stderr`.
`run(source, check=True)` is `node --check`: parse, do not run.

It writes no file, so two workers cannot clobber each other's scratch script, which the
per-check helpers did with fixed names under `tests/`.

A context is not a process. A check cannot leak globals into the next one, but a check that
never lets go -- an interval nobody clears -- would hang this node where it would have hung
its own; `pytest-timeout` is the guard in both cases.

**What the node holds, and for how long** (Q749). V8's compilation cache keeps every
script it compiles, keyed by source, and a script compiled into a `vm` context keeps that
context: so each check's context, with the 1.3 MB page script in it, lived as long as the
node. Asked by `test_web.py` alone, the node grew to 1.0 GB, and the four a fast run keeps
held 2.0-2.3 GB at its peak (F4's after, `docs/archive/plan-tests.md`). Started with
`--no-compilation-cache` it holds about 160 MB on the same file, for no measurable time: every
check compiles into a fresh context anyway. `--max-old-space-size` is the backstop, so a leak
found later costs a crash naming this file rather than a gigabyte per worker. And the node is
released when the last test of a module that used it ends (`pytest_runtest_teardown` below,
this file being a plugin `conftest.py` names), so a worker in the rest of the suite holds none;
the next check starts another, at about a tenth of a second.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

# The heap a check may use. A check's live heap is tens of megabytes; this is a ceiling a
# leak hits, well above what any check needs. The `suite` lease's booking (`SUITE_PEAK_MB` in
# `.claude/scripts/compute_lease.py`) was measured with each node at 100-125 MB.
HEAP_MB = 256
NODE_FLAGS = ("--no-compilation-cache", f"--max-old-space-size={HEAP_MB}")

# The server half. It reads one JSON job per line and answers one JSON line per job, in
# order. Everything a job prints goes into its own buffers, never to this process's stdout,
# which is the channel.
_SERVER = r"""
'use strict';
const vm = require('vm');
const util = require('util');
const { Console } = require('console');
const { Writable } = require('stream');
const readline = require('readline');

const SPECIAL = new Set(['console', 'process', 'global', 'globalThis', 'setTimeout',
  'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate', 'clearImmediate',
  'queueMicrotask']);
// What node adds to a script's globals beyond the language's own (`URL`, `performance`,
// `structuredClone`, `fetch`, ...). A fresh context already has its own `Object`, `Array`
// and the rest, and must keep them: a host `Object` in there would fail `instanceof`.
const own = new Set(vm.runInNewContext('Object.getOwnPropertyNames(globalThis)'));
const hostNames = Object.getOwnPropertyNames(globalThis)
  .filter(name => !SPECIAL.has(name) && !own.has(name));

let current = null;
process.on('unhandledRejection', reason => { if (current) current.fail(reason); });
process.on('uncaughtException', error => { if (current) current.fail(error); });

function sink(into) {
  return new Writable({ write(chunk, encoding, done) { into.push(String(chunk)); done(); } });
}

function describe(error) {
  if (error && error.stack) return String(error.stack);
  return 'Uncaught ' + util.inspect(error);
}

async function runJob(job) {
  const out = [], err = [];
  const state = { failed: null, exit: null, pending: new Map(), wake: null };
  const fail = error => {
    if (state.failed === null && state.exit === null) state.failed = error;
    for (const [handle, clear] of state.pending) clear(handle);
    state.pending.clear();
    if (state.wake) state.wake();
  };
  current = { fail };
  const EXIT = Symbol('exit');
  const guard = fn => function (...args) {
    try { return fn.apply(this, args); }
    catch (error) { if (error !== EXIT) fail(error); }
  };
  const settled = handle => {
    state.pending.delete(handle);
    if (state.wake) state.wake();
  };
  const timer = (start, clear, repeats) => (fn, ...rest) => {
    let handle;
    const call = guard((...args) => { if (!repeats) settled(handle); return fn(...args); });
    handle = start(call, ...rest);
    state.pending.set(handle, clear);
    return handle;
  };
  const clearing = clear => handle => {
    if (state.pending.has(handle)) settled(handle);
    clear(handle);
  };

  const sandbox = {};
  for (const name of hostNames) {
    try { sandbox[name] = globalThis[name]; } catch (e) { /* a getter that refuses */ }
  }
  sandbox.console = new Console({ stdout: sink(out), stderr: sink(err), colorMode: false });
  sandbox.setTimeout = timer(setTimeout, clearTimeout, false);
  sandbox.setInterval = timer(setInterval, clearInterval, true);
  sandbox.setImmediate = timer(setImmediate, clearImmediate, false);
  sandbox.clearTimeout = clearing(clearTimeout);
  sandbox.clearInterval = clearing(clearInterval);
  sandbox.clearImmediate = clearing(clearImmediate);
  sandbox.queueMicrotask = fn => queueMicrotask(guard(fn));
  sandbox.process = {
    env: {}, argv: ['node', 'check.js'], platform: process.platform,
    version: process.version, versions: process.versions,
    hrtime: process.hrtime, nextTick: (fn, ...args) => queueMicrotask(guard(() => fn(...args))),
    stdout: { write: text => { out.push(String(text)); return true; } },
    stderr: { write: text => { err.push(String(text)); return true; } },
    exit: code => { state.exit = code === undefined ? 0 : code; throw EXIT; },
  };
  sandbox.global = sandbox;
  const context = vm.createContext(sandbox);

  let body;
  try {
    body = vm.compileFunction(job.source,
      ['exports', 'require', 'module', '__filename', '__dirname'],
      { parsingContext: context, filename: 'check.js' });
  } catch (error) {
    return { returncode: 1, stdout: '', stderr: describe(error) };
  }
  if (!job.check) {
    const module = { exports: {} };
    try {
      body.call(module.exports, module.exports, require, module, 'check.js', '.');
    } catch (error) {
      if (error !== EXIT) fail(error);
    }
    // Wait until nothing the job scheduled is left, draining microtasks in between: the
    // moment `node file.js` would have exited.
    for (;;) {
      await new Promise(resolve => setImmediate(resolve));
      if (state.failed !== null || state.exit !== null || state.pending.size === 0) break;
      await new Promise(resolve => { state.wake = resolve; });
      state.wake = null;
    }
    for (const [handle, clear] of state.pending) clear(handle);
  }
  current = null;
  let returncode = 0;
  if (state.exit !== null) returncode = state.exit;
  else if (state.failed !== null) { returncode = 1; err.push(describe(state.failed) + '\n'); }
  return { returncode, stdout: out.join(''), stderr: err.join('') };
}

const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let queue = Promise.resolve();
lines.on('line', line => {
  const job = JSON.parse(line);
  queue = queue.then(async () => {
    let answer;
    try { answer = await runJob(job); }
    catch (error) { answer = { returncode: 1, stdout: '', stderr: describe(error) }; }
    process.stdout.write(JSON.stringify(answer) + '\n');
  });
});
lines.on('close', () => queue.then(() => process.exit(0)));
"""


@dataclass(frozen=True)
class Ran:
    """What `subprocess.run([node, file])` returned, as far as any helper reads it."""

    returncode: int
    stdout: str
    stderr: str


class _Node:
    def __init__(self, node: str) -> None:
        self.lock = threading.Lock()
        self.process = subprocess.Popen(
            [node, *NODE_FLAGS, "-e", _SERVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            bufsize=1,
        )

    def ask(self, source: str, check: bool) -> Ran:
        assert self.process.stdin is not None and self.process.stdout is not None
        with self.lock:
            self.process.stdin.write(json.dumps({"source": source, "check": check}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            code = self.process.wait()
            raise RuntimeError(
                f"the session's node exited ({code}); a check that needs more than"
                f" {HEAP_MB} MB of heap ends it, so see HEAP_MB in tests/node_session.py"
            )
        answer = json.loads(line)
        return Ran(answer["returncode"], answer["stdout"], answer["stderr"])

    def close(self) -> None:
        if self.process.poll() is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


_node: _Node | None = None

# Every answer, by the exact source it was for. The checks are deterministic -- a stubbed
# DOM, a `fetch` that answers from a table or never, no clock anyone prints -- and several
# tests ask the same one: six `TestAnErrorLandsWhereTheReaderIsLooking` tests read one
# `_error_paths()` run, and the game-page helpers are called with the same arguments from
# several tests. A `Ran` is frozen, so one can be handed to each.
_answers: dict[tuple[str, bool], Ran] = {}


def run(source: str, *, check: bool = False) -> Ran:
    """Run `source` as `node file.js` would, or only parse it (`node --check`) with `check`.

    Skips the calling test where node is not installed, as every helper did. The same
    source asked twice is answered from the first run.
    """
    global _node
    key = (hashlib.sha256(source.encode("utf-8")).hexdigest(), check)
    answer = _answers.get(key)
    if answer is not None:
        return answer
    if _node is None or _node.process.poll() is not None:
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")
        _node = _Node(node)
        atexit.register(_node.close)
    answer = _answers[key] = _node.ask(source, check)
    return answer


def release() -> bool:
    """End the session's node, if one is running; the next `run` starts another.

    True when there was one to end. The answers already given are kept.
    """
    global _node
    if _node is None:
        return False
    held, _node = _node, None
    held.close()
    return True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """When a module's last test (on this worker) is torn down, its node goes with it.

    After the teardown, so a module-scoped fixture that asks node in its finaliser still
    has one. The next module that asks starts its own: a worker switches out of a module
    that used node 4-10 times in a fast run (Q749), at about a tenth of a second each.
    """
    yield
    if nextitem is None or nextitem.path != item.path:
        release()
