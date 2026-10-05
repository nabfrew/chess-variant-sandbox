// The browser half of tests/test_browser_identity.py: the artefact's wheel in Pyodide
// under Node, running identity_probe.py on one job.
//
//   node identity.mjs <runtime dir> <wheel> <identity_probe.py>  < job.json  > reply.json
//
// The test runs this inside the image deploy/pyodide/build.sh tagged, on the Node the
// wheel was built beside, with build/browser/ and this directory mounted read-only.
//
// Like deploy/pyodide/verify.mjs, it holds no position and answers no question. It reads
// the job off stdin as text, hands that text to Python, and writes Python's reply to
// stdout. JavaScript parses neither: records carry 64-bit position keys, and a
// JavaScript number is exact to 53 bits. Pyodide's own chatter goes to stderr, so stdout
// carries the reply and nothing else.

import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const [runtimeDir, wheelPath, probePath] = process.argv.slice(2);
if (!runtimeDir || !wheelPath || !probePath) {
    process.stderr.write("usage: node identity.mjs <runtime dir> <wheel> <identity_probe.py>\n");
    process.exit(2);
}

const chunks = [];
for await (const chunk of process.stdin) {
    chunks.push(chunk);
}
const job = Buffer.concat(chunks).toString("utf8");

// Everything but the reply goes to stderr: Python's prints, and whatever Pyodide says
// through `console.log` while it loads (`loadPackage` announces each package there).
const toStderr = (...parts) => process.stderr.write(`${parts.join(" ")}\n`);
console.log = toStderr;
console.info = toStderr;
console.debug = toStderr;

const runtime = resolve(runtimeDir);
const { loadPyodide } = await import(pathToFileURL(join(runtime, "pyodide.mjs")).href);
const pyodide = await loadPyodide({ indexURL: runtime, stdout: toStderr, stderr: toStderr });

// The wheel under test, by path, with no micropip -- what a browser worker does (B1).
await pyodide.loadPackage(resolve(wheelPath), {
    messageCallback: toStderr,
    errorCallback: toStderr,
});

// The probe goes in as one file, loaded by path. The tree is never put on sys.path,
// because `chess_variant_sandbox` has to be the wheel's and not the working tree's.
pyodide.FS.mkdirTree("/probe");
pyodide.FS.writeFile("/probe/identity_probe.py", readFileSync(resolve(probePath)));

pyodide.globals.set("identity_job", job);
pyodide.globals.set(
    "identity_host",
    JSON.stringify({ node: process.versions.node, pyodide: pyodide.version }),
);

const reply = await pyodide.runPythonAsync(`
import importlib.util, json, sys

location = importlib.util.spec_from_file_location("identity_probe", "/probe/identity_probe.py")
probe = importlib.util.module_from_spec(location)
sys.modules[location.name] = probe
location.loader.exec_module(probe)

probe.run(identity_job, json.loads(identity_host))
`);

process.stdout.write(reply);
