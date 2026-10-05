"""Check that every icon in the library actually draws inside its own frame.

    python assets/pieces/check.py

Half the source files declare no ``viewBox``, so ``fetch.py`` derives one from
``width``/``height``. That is right only if the art was authored at the origin,
and nothing else in the pipeline would notice if it were not: the files parse,
the licences check out, the manifest looks sane, and the page still renders
something. A file that parses is not a file that draws. This measures what a
browser actually paints -- the bounding box of the art against the frame it is
supposed to fill -- and flags anything clipped, empty, or lost in a corner.

It is deliberately a separate script rather than part of ``fetch.py``: it needs a
browser, and the fetch has to work on a machine without one.

Needs a Chrome or Edge binary. Without one it reports that it could not check
rather than passing silently.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
]

# Fraction of the frame the art must cover, and how far outside it may stray.
# The bounds are loose: they are meant to catch art that is clipped or invisible,
# not to police composition.
MIN_COVERAGE = 0.25
MAX_OVERFLOW = 0.02

PROBE = """<meta charset="utf-8"><body><script>
const SOURCES = %s;
const FILES = Object.keys(SOURCES);
async function bbox(file) {
  const text = SOURCES[file];
  const host = document.createElement('div');
  host.style.cssText = 'position:absolute;left:0;top:0;width:400px;height:400px';
  host.innerHTML = text;
  document.body.appendChild(host);
  const svg = host.querySelector('svg');
  if (!svg) { host.remove(); return {error: 'no <svg> element'}; }
  const vb = svg.getAttribute('viewBox');
  let frame;
  if (vb) {
    const n = vb.split(/[\\s,]+/).map(Number);
    frame = {x: n[0], y: n[1], w: n[2], h: n[3]};
  } else {
    frame = {x: 0, y: 0, w: parseFloat(svg.getAttribute('width')),
             h: parseFloat(svg.getAttribute('height'))};
  }
  let box;
  try { box = svg.getBBox(); } catch (e) { host.remove(); return {error: String(e)}; }
  host.remove();
  return {frame, box: {x: box.x, y: box.y, w: box.width, h: box.height}};
}
(async () => {
  const out = {};
  for (const f of FILES) {
    try { out[f] = await bbox(f); } catch (e) { out[f] = {error: String(e)}; }
  }
  document.title = 'DONE';
  const pre = document.createElement('pre');
  pre.id = 'result';
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
})();
</script></body>"""


def find_browser() -> Path | None:
    return next((p for p in BROWSERS if p.exists()), None)


def measure(browser: Path, files: list[str]) -> dict:
    # The sources are embedded rather than fetched: a file:// page cannot fetch
    # siblings, and inlining keeps the probe independent of where it is written.
    sources = {f: (ROOT / f).read_text(encoding="utf-8", errors="replace") for f in files}
    payload = json.dumps(sources).replace("</", "<\\/")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "probe.html"
        page.write_text(PROBE % payload, encoding="utf-8")
        # --dump-dom runs the page and prints the settled DOM, which carries the
        # <pre> the probe fills in.
        proc = subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--virtual-time-budget=15000",
                "--allow-file-access-from-files",
                "--dump-dom",
                page.as_uri(),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
    html = proc.stdout
    start = html.find('<pre id="result">')
    if start < 0:
        raise SystemExit("probe did not produce a result; browser said:\n" + proc.stderr[-2000:])
    start = html.index(">", start) + 1
    end = html.index("</pre>", start)
    return json.loads(html[start:end].replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))


def main() -> int:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    browser = find_browser()
    if browser is None:
        print("no Chrome or Edge found; cannot check rendering", file=sys.stderr)
        return 2

    files = [entry["file"] for entry in manifest.values()]
    by_file = {entry["file"]: slug for slug, entry in manifest.items()}
    results = measure(browser, files)

    problems: list[str] = []
    for file, res in sorted(results.items()):
        slug = by_file.get(file, file)
        if "error" in res:
            problems.append(f"{slug:<32} {res['error']}")
            continue
        frame, box = res["frame"], res["box"]
        if not frame["w"] or not frame["h"]:
            problems.append(f"{slug:<32} frame has no size")
            continue
        if box["w"] <= 0 or box["h"] <= 0:
            problems.append(f"{slug:<32} draws nothing")
            continue
        coverage = (box["w"] * box["h"]) / (frame["w"] * frame["h"])
        over = max(
            frame["x"] - box["x"],
            frame["y"] - box["y"],
            (box["x"] + box["w"]) - (frame["x"] + frame["w"]),
            (box["y"] + box["h"]) - (frame["y"] + frame["h"]),
        )
        overflow = over / max(frame["w"], frame["h"])
        if overflow > MAX_OVERFLOW:
            problems.append(
                f"{slug:<32} art falls {overflow * 100:.0f}% outside its frame (clipped)"
            )
        elif coverage < MIN_COVERAGE:
            problems.append(
                f"{slug:<32} art fills only {coverage * 100:.0f}% of its frame"
            )

    print(f"checked {len(results)} icons with {browser.name}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print("  " + line)
        return 1
    print("all icons draw inside their frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
