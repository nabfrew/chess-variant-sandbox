// The browser's measuring worker: one unit of a measurement in, its record text out.
//
// `docs/plan-client-compute.md` §8, stage B3. A measurement is cut into units by
// `chess_variant_sandbox/tournament/slices.py` (B0), and a unit is one request text that
// holds the spec, the unit's games by schedule index, the fit's verdict, the roster and,
// for a reading phase, the records it reads. This worker loads Pyodide, installs the
// package's wheel, and hands each request to `slices.run_request`, which is the whole of
// what a worker does. The core that plays is the package's own `_core`, built for
// Pyodide's Emscripten target. The one worker a page's measurement gathers in (B5) also
// runs the coordinator's gathering half between the phases, `slices.gather_request`,
// which is in the same wheel. So every line of Python a worker runs came in the wheel the
// manifest names, and none of it came from the page (Q458).
//
// So this file holds no position and answers no question. It never computes a legal
// move, an outcome, a move choice, a schedule, a seed or a score, and it never parses a
// request or a reply: records carry 64-bit position keys, and a JavaScript number is
// exact to 53 bits. It moves text between its host and Python, and it measures two
// things that are about the machine and not the game -- the size of the WebAssembly heap
// and the wall clock.
//
// One file, two hosts. A browser starts it as a worker, classic or module. Node starts it
// under `worker_threads` through `tests/browser/node-worker.mjs`, a few lines that give
// Node a worker's `self`, `postMessage`, `onmessage` and `close`, so the suite runs the
// file that ships.
//
// Messages in, handled strictly in the order they arrive:
//
//   {kind: "load", runtime, wheel, gathers}
//       `runtime` is the URL of the Pyodide directory the manifest names
//       (`/browser/pyodide-<version>/`), and Pyodide finds its other files beside
//       `pyodide.mjs` there. `wheel` is the wheel as `pyodide.loadPackage` takes it: its
//       URL in a browser, and its file path under Node, where Pyodide reads the file
//       with `fs` and cannot read a `file:` URL on Windows. In a browser both may be
//       relative to the worker's own URL.
//       `gathers` is true only for the one worker that gathers (B5). That worker also
//       takes `slices.gather_request` from the wheel, and a wheel without it -- one
//       built before Q458 -- is refused here by name, before any unit is dealt.
//   {kind: "unit", id, request}
//       `request` is `slices.request(...)`'s text, and `id` is the sender's name for it.
//   {kind: "gather", id, request}
//       One step of the gathering half: `request` is the page's text for
//       `slices.gather_request`.
//   {kind: "close"}
//       The worker ends itself, through the worker's own `self.close()`, once everything
//       sent before it has been answered. Nothing is posted back.
//
// Messages out:
//
//   {kind: "loaded", pyodide, heap, seconds}
//   {kind: "unit", id, text, heap, seconds}
//       `text` is `slices.run_request`'s reply, JSON Lines, exactly as Python wrote it.
//   {kind: "gathered", id, text, heap, seconds}
//       `text` is `slices.gather_request`'s answer, exactly as Python wrote it.
//   {kind: "error", id, message}
//       `id` is the unit's, or null for a load. The worker stays up for the next unit.
//
// `heap` is the size of the WebAssembly module's linear memory in bytes. It only ever
// grows, so the number posted after a unit is the high-water mark of everything this
// worker has run so far.

"use strict";

let pyodide = null;
let runRequest = null;
let gatherStep = null;

function heapBytes() {
    // Pyodide keeps Emscripten's module as `_module`, and `HEAP8` is re-made over the
    // new buffer each time the memory grows.
    const heap = pyodide && pyodide._module && pyodide._module.HEAP8;
    return heap ? heap.buffer.byteLength : null;
}

function seconds(started) {
    return Math.round(performance.now() - started) / 1000;
}

async function load(message) {
    if (pyodide !== null) {
        throw new Error("this worker has loaded Pyodide already");
    }
    const started = performance.now();
    // Relative to the worker's own URL in a browser; Node's worker has no location, and
    // is handed an absolute `file:` URL.
    const where = self.location ? self.location.href : undefined;
    const base = new URL(String(message.runtime), where).href;
    const module = new URL("pyodide.mjs", base.endsWith("/") ? base : `${base}/`);
    const { loadPyodide } = await import(module.href);
    // No `indexURL`: Pyodide then takes the directory `pyodide.mjs` came from, which is
    // the base above in a browser and under Node alike. Python's own output goes to the
    // console, which is Pyodide's default in a browser; under Node its default writes to
    // `process.stdout.fd`, and a worker thread has none.
    const loaded = await loadPyodide({
        stdout: (line) => console.log(line),
        stderr: (line) => console.warn(line),
    });
    await loaded.loadPackage(String(message.wheel));
    const slices = loaded.pyimport("chess_variant_sandbox.tournament.slices");
    if (message.gathers === true) {
        // A wheel's module answers `undefined` for a name it does not have.
        const step = slices.gather_request;
        if (step === undefined || step === null) {
            throw new Error(
                "this wheel has no slices.gather_request, so it cannot gather: it was built "
                + "before Q458, and build/browser/ wants rebuilding (deploy/pyodide/build.sh)");
        }
        gatherStep = step;
    }
    runRequest = slices.run_request;
    pyodide = loaded;
    return { kind: "loaded", pyodide: pyodide.version, heap: heapBytes(), seconds: seconds(started) };
}

async function unit(message) {
    if (runRequest === null) {
        throw new Error("a unit arrived before Pyodide was loaded");
    }
    const started = performance.now();
    const text = runRequest(String(message.request));
    return { kind: "unit", id: message.id, text, heap: heapBytes(), seconds: seconds(started) };
}

async function gather(message) {
    if (gatherStep === null) {
        throw new Error("this worker was loaded without the gathering half");
    }
    const started = performance.now();
    const text = gatherStep(String(message.request));
    return { kind: "gathered", id: message.id, text, heap: heapBytes(), seconds: seconds(started) };
}

async function close() {
    self.close();
    return null;
}

const HANDLERS = { load, unit, gather, close };

async function handle(message) {
    const id = message && message.id !== undefined ? message.id : null;
    try {
        const handler = message && Object.hasOwn(HANDLERS, message.kind) ? HANDLERS[message.kind] : null;
        if (handler === null) {
            throw new Error(`no such message kind: ${message && message.kind}`);
        }
        const answer = await handler(message);
        if (answer !== null) {
            self.postMessage(answer);
        }
    } catch (error) {
        self.postMessage({ kind: "error", id, message: String((error && error.message) || error) });
    }
}

// One at a time, in arrival order: a unit posted while Pyodide is still loading waits
// for the load, and two units never share the interpreter.
let queue = Promise.resolve();
self.onmessage = (event) => {
    queue = queue.then(() => handle(event.data));
};
