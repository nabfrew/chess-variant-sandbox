// The page's side of `chess_variant_sandbox/web/static/measure-worker.js`, under Node, for
// `tests/test_measure_worker.py` (B3).
//
//   node worker-host.mjs <worker script> <runtime URL> <wheel> <request file>...  > reply.json
//
// It starts the worker through `node-worker.mjs` and sends it one `load`, one `unit` a
// request file in order, and a `close`. It prints one JSON object: the `loaded` message
// and every unit's message, as the worker posted them. It exits 0 when every message was
// a success and 3 when any was an `error`, which it prints as well.
//
// Like the worker, it holds no position and answers no question. A request file is read
// as text and posted as text, and a unit's reply goes out inside `JSON.stringify`, which
// writes a string back exactly; nothing here parses a record. What the worker prints goes
// to stderr, so stdout carries the reply and nothing else.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { finished } from "node:stream/promises";
import { pathToFileURL } from "node:url";
import { Worker } from "node:worker_threads";

// How long a worker that has answered everything may take to end on its own.
const CLOSE_MS = 30000;

const [script, runtime, wheel, ...requests] = process.argv.slice(2);
if (!script || !runtime || !wheel) {
    process.stderr.write(
        "usage: node worker-host.mjs <worker script> <runtime URL> <wheel> <request file>...\n",
    );
    process.exit(2);
}

const worker = new Worker(new URL("./node-worker.mjs", import.meta.url), {
    workerData: { script: pathToFileURL(resolve(script)).href },
    stdout: true,
    stderr: true,
});
worker.stdout.pipe(process.stderr);
worker.stderr.pipe(process.stderr);

// One answer to the load and one to each unit, errors included, and then the worker ends.
const expected = requests.length + 1;
const messages = [];
let closing = null;
// This process's resident set when the last answer arrives, while the worker is still up:
// the heap is the module's linear memory alone, and this is everything a worker costs,
// Node's own share included.
let rss = null;
const ended = new Promise((finish, fail) => {
    worker.on("error", fail);
    worker.on("exit", finish);
    worker.on("message", (message) => {
        messages.push(message);
        if (messages.length === expected) {
            rss = process.memoryUsage().rss;
            closing = setTimeout(() => {
                process.stderr.write(`the worker was still up ${CLOSE_MS} ms after closing\n`);
                worker.terminate();
            }, CLOSE_MS);
        }
    });
});

worker.postMessage({ kind: "load", runtime, wheel });
requests.forEach((path, id) => {
    worker.postMessage({ kind: "unit", id, request: readFileSync(resolve(path), "utf8") });
});
worker.postMessage({ kind: "close" });

// The worker's console reaches this thread on a channel of its own, which can trail its
// last message; a worker that ends on its own has handed all of it over first.
const code = await ended;
clearTimeout(closing);
await Promise.all([finished(worker.stdout), finished(worker.stderr)]);

const [loaded, ...units] = messages;
process.stdout.write(`${JSON.stringify({ node: process.versions.node, rss, loaded, units })}\n`);
if (messages.length !== expected) {
    process.stderr.write(`the worker exited with ${code} after ${messages.length} of ${expected} answers\n`);
    process.exitCode = 4;
} else {
    process.exitCode = messages.some((message) => message.kind === "error") ? 3 : 0;
}
