// Node's `worker_threads` dressed as a browser worker's global scope, so that
// `chess_variant_sandbox/web/static/measure-worker.js` runs here as it ships (B3).
//
//   new Worker("node-worker.mjs", { workerData: { script: <file URL of the worker> } })
//
// A browser worker has `self`, `self.postMessage`, `self.onmessage` and `self.close`, and
// Node's has a `parentPort`. These lines are the whole difference. The worker is imported
// before the port is listened to, and a port queues what arrives until then, so nothing
// is lost. Closing the port lets the thread end on its own, once its console has been
// handed to the parent.

import { parentPort, workerData } from "node:worker_threads";

globalThis.self = globalThis;
globalThis.postMessage = (message) => parentPort.postMessage(message);
globalThis.close = () => parentPort.close();
await import(workerData.script);
parentPort.on("message", (data) => globalThis.onmessage({ data }));
