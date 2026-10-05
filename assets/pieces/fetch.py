"""Rebuild assets/pieces/ from Wikimedia Commons.

The library is committed, so this script is provenance rather than a build step:
it records exactly which Commons file every icon came from, re-checks the licence
of each one against an allowlist, and regenerates CREDITS.md and manifest.json.

Run from anywhere:  python assets/pieces/fetch.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "chess-variant-sandbox-piece-icons/1.0 (research harness) python-urllib"

# Seconds between file downloads. The file host tolerates a short burst and then
# bans the IP for several minutes; a steady slow pace never trips it.
THROTTLE = 8.0
ROOT = Path(__file__).resolve().parent
SVG_DIR = ROOT / "svg"

# Licences we are willing to redistribute. Anything outside this set is a hard
# failure rather than a silent skip: a piece quietly vanishing from the library
# is worse than a rebuild that stops and says why.
ALLOWED_LICENCES = {
    "CC0",
    "CC BY 3.0",
    "CC BY 4.0",
    "CC BY 3.0 de",
    "CC BY-SA 3.0",
    "CC BY-SA 4.0",
    "Public domain",
    "GFDL",
    "OFL",
}

# slug -> Commons letter in the "Chess <letter><colour>t45.svg" family.
# Two-colour western pieces, 45x45, transparent background, one consistent style.
CBURNETT = {
    # the six a chess-like game almost always needs
    "king": "k",
    "queen": "q",
    "rook": "r",
    "bishop": "b",
    "knight": "n",
    "pawn": "p",
    # compounds
    "archbishop": "a",
    "chancellor": "c",
    "amazon": "A",
    "centaur": "C",
    # short-range steppers and leapers
    "ferz": "F",
    "dabbaba": "v",
    "man": "M",
    "commoner": "x",
    # long leapers.  Not "N": that file's art falls outside its own 45x45 canvas
    # and renders clipped.  "K" is the same piece, drawn like the G/Z/U leapers.
    "nightrider": "K",
    "zebra": "Z",
    "giraffe": "G",
    "unicorn": "U",
    # the rest of the fairy vocabulary
    "dragon": "D",
    "champion": "z",
    # Commons calls this one "wizard"; the drawing is unambiguously a crescent,
    # so the library calls it what it looks like.
    "moon": "w",
    "ship": "s",
    "short-rook": "S",
    # inverted forms, useful for promoted pieces or a rotated second player
    "inverted-king": "f",
    "inverted-queen": "g",
    "inverted-pawn": "h",
}

# Two-colour pieces that do not follow the Chess_<x><c>t45 naming.
PAIRS = {
    "alfil": ("File:White alfil.svg", "File:Black alfil.svg"),
    "camel": ("File:White chess camel.svg", "File:Black chess camel.svg"),
    # Not Chess_W{l,d}t45: the light one is a vector but the dark one is a PNG
    # in an SVG wrapper, so the pair does not match. These two are Quivira
    # glyphs, the same family as alfil and camel above.
    "wazir": ("File:White wazir.svg", "File:Black wazir.svg"),
}

# Single-colour glyphs. Shogi and xiangqi distinguish the two players by
# rotating the piece, not by recolouring it, so one file serves both seats.
SINGLES = {
    "shogi-" + slug: "File:Shogi " + name + " western.svg"
    for slug, name in [
        ("king", "king"),
        ("rook", "rook"),
        ("bishop", "bishop"),
        ("gold", "gold general"),
        ("silver", "silver"),
        ("knight", "knight"),
        ("lance", "lance"),
        ("pawn", "pawn"),
        ("promoted-rook", "promoted rook"),
        ("promoted-bishop", "promoted bishop"),
        ("promoted-silver", "promoted silver"),
        ("promoted-knight", "promoted knight"),
        ("promoted-lance", "promoted lance"),
        ("promoted-pawn", "promoted pawn"),
    ]
}
# Icons drawn for this repository rather than downloaded, because Commons has no
# picture of the thing. Rock, paper and scissors are pieces in `rulesets/rps-*.json`
# and wore a dabbaba, a crescent and a giraffe until they were drawn -- three fairy
# glyphs picked for being distinguishable from each other, which is a placeholder and
# not a drawing. They are authored in the Cburnett geometry (45x45, #ffffff or #000000
# fill, a 1.5 black stroke) so they sit on a board next to that set without looking
# borrowed, and released CC0 so they carry no obligation onto anything that ships them.
#
# `main()` never downloads these and never overwrites them; it reads what is on disk
# and records it, which is the only reason a rebuild does not silently drop them from
# `manifest.json` and from CREDITS.md.
#
# They have no `source`: the file here is the original, and the credits cite no
# repository address (the user's call, 2026-09-24).
LOCAL = {
    stem + "-" + colour: description
    for stem, description in [
        ("rock", "a faceted boulder"),
        ("paper", "a sheet with a folded corner"),
        ("scissors", "an open pair of scissors"),
    ]
    for colour in ("white", "black")
}
LOCAL_CREDIT = {
    "author": "Chess variant sandbox",
    "licence": "CC0",
    "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
}

# Xiangqi arrives four-up: one file per piece holding the red character, a red
# pictogram, the black character and the black pictogram, side by side on a
# 400x100 strip. split_xiangqi() cuts each into its own file.
XIANGQI = {
    "general": "File:Xiangqi General (Trad).svg",
    "advisor": "File:Xiangqi Advisor (Trad).svg",
    "elephant": "File:Xiangqi Elephant (Trad).svg",
    "horse": "File:Xiangqi Horse (Trad).svg",
    "chariot": "File:Xiangqi Chariot (Trad).svg",
    "cannon": "File:Xiangqi Cannon (Trad).svg",
    "soldier": "File:Xiangqi Soldier (Trad).svg",
}

# Offset of each disc on the strip -> (stem prefix, colour suffix). The two
# character forms differ by side, which is why red and black are not one file
# rotated: xiangqi names the same piece with different characters per player.
XIANGQI_CUTS = {
    -150: ("xiangqi-{piece}", "white"),
    -50: ("xiangqi-pictorial-{piece}", "white"),
    50: ("xiangqi-{piece}", "black"),
    150: ("xiangqi-pictorial-{piece}", "black"),
}


def split_xiangqi(piece: str, blob: bytes) -> dict[str, bytes]:
    """Cut one four-up strip into four centred 100x100 files."""
    text = blob.decode("utf-8")
    note = re.search(r"<!--.*?-->", text, re.S)
    groups = dict(
        (int(x), body)
        for x, body in re.findall(
            r'<g transform="translate\((-?\d+),0\)">(.*?)</g>', text, re.S
        )
    )
    missing = set(XIANGQI_CUTS) - set(groups)
    if missing:
        raise SystemExit(f"xiangqi {piece}: no disc at {sorted(missing)}")
    out: dict[str, bytes] = {}
    for offset, (pattern, colour) in XIANGQI_CUTS.items():
        stem = pattern.format(piece=piece) + "-" + colour
        # The disc is drawn centred on its group origin with r=48.5 and a 3-wide
        # stroke, so dropping the translate and framing -50..50 centres it exactly.
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg width="100" height="100" viewBox="-50 -50 100 100"',
            '     xmlns="http://www.w3.org/2000/svg">',
            "  <title>" + stem + "</title>",
        ]
        if note:
            parts.append("  " + note.group(0))
        parts.append("  " + groups[offset].strip())
        parts.append("</svg>")
        out[stem] = ("\n".join(parts) + "\n").encode("utf-8")
    return out


def api(**params: str) -> dict:
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    body = urllib.parse.urlencode(params).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(API, data=body, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as fh:
                return json.load(fh)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise SystemExit("Commons API kept refusing; try again later")


def strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def fetch_metadata(titles: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 40):
        data = api(
            action="query",
            titles="|".join(titles[i : i + 40]),
            prop="imageinfo",
            iiprop="url|extmetadata|mime|size",
            iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|ImageDescription",
        )
        for page in data["query"]["pages"]:
            if "imageinfo" not in page:
                raise SystemExit("no such file on Commons: " + str(page.get("title")))
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})

            def get(key: str, _meta: dict = meta) -> str:
                return strip_html(_meta.get(key, {}).get("value", ""))

            out[page["title"]] = {
                "url": info["url"],
                "page": info["descriptionurl"],
                "mime": info["mime"],
                "size": info["size"],
                "licence": get("LicenseShortName"),
                "licence_url": get("LicenseUrl"),
                "artist": get("Artist"),
                "description": get("ImageDescription"),
            }
        time.sleep(1.2)
    return out


def download(url: str) -> bytes:
    # The API decorates file URLs with utm_* tracking parameters; upload.wikimedia.org
    # rate-limits the decorated form far more aggressively than the bare one.
    url = urllib.parse.urldefrag(url.split("?", 1)[0]).url
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as fh:
                return fh.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503):
                # upload.wikimedia.org bans by IP for minutes at a time, and every
                # request made while banned restarts the clock. Wait it out rather
                # than probing.
                time.sleep(300 + 120 * attempt)
                continue
            raise
    raise SystemExit("upload.wikimedia.org kept refusing: " + url)


# These files are third-party markup that ends up inside a generated HTML page.
# A reference that leaves the document -- anything but a same-document "#id" or an
# inline data: URI -- would make the page fetch from somewhere at render time.
UNSAFE = re.compile(
    rb"<\s*script"
    rb"|<\s*foreignObject"
    rb"|\son\w+\s*="
    rb"|javascript:"
    rb"|<!ENTITY"
    rb"|(?:xlink:)?href\s*=\s*[\"'](?!\s*(?:#|data:))",
    re.IGNORECASE,
)

# An SVG that is really a base64 bitmap in a wrapper. Free, valid, and useless
# as an icon: it will not scale, and it will not recolour.
RASTER = re.compile(rb"data:image/(?!svg)", re.IGNORECASE)


def check_safe(name: str, blob: bytes) -> None:
    match = UNSAFE.search(blob)
    if match:
        raise SystemExit(name + ": refusing, contains " + repr(match.group(0)))
    if RASTER.search(blob):
        raise SystemExit(name + ": refusing, an embedded bitmap rather than a vector")


def view_box(blob: bytes) -> str:
    text = blob.decode("utf-8", "replace")
    found = re.search(r'\bviewBox\s*=\s*"([^"]+)"', text)
    if found:
        return " ".join(found.group(1).split())
    width = re.search(r'<svg[^>]*\bwidth\s*=\s*"([\d.]+)', text)
    height = re.search(r'<svg[^>]*\bheight\s*=\s*"([\d.]+)', text)
    if width and height:
        return "0 0 " + width.group(1) + " " + height.group(1)
    raise SystemExit("svg has neither viewBox nor numeric width/height")


def write_credits(manifest: dict[str, dict]) -> None:
    lines = [
        "# Credits",
        "",
        "Every file in `svg/` is under the licence named below. Most came from",
        "Wikimedia Commons and are redistributed byte-for-byte unmodified; the",
        "xiangqi files are each one disc cut out of a four-up source sheet, and the",
        "six rock/paper/scissors files were drawn for this repository and released",
        "CC0. The notes column says which is which. Generated by `fetch.py`; edit",
        "that script, not this file.",
        "",
        "| file | author | licence | source | notes |",
        "|---|---|---|---|---|",
    ]
    for stem, entry in sorted(manifest.items()):
        author = entry["author"].replace("|", "/") or "unknown"
        lic = entry["licence"]
        if entry["licence_url"]:
            lic = "[" + lic + "](" + entry["licence_url"] + ")"
        # A file drawn here has no source to link: it is the original.
        source = "[Commons](" + entry["source"] + ")" if entry.get("source") else "original"
        lines.append(
            "| `" + stem + ".svg` | " + author + " | " + lic + " | " + source + " | "
            + entry.get("modified", "") + " |"
        )
    lines.append("")
    (ROOT / "CREDITS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    wanted: dict[str, str] = {}  # output stem -> Commons title
    for slug, letter in CBURNETT.items():
        wanted[slug + "-white"] = "File:Chess " + letter + "lt45.svg"
        wanted[slug + "-black"] = "File:Chess " + letter + "dt45.svg"
    for slug, (light, dark) in PAIRS.items():
        wanted[slug + "-white"] = light
        wanted[slug + "-black"] = dark
    wanted.update(SINGLES)

    print("resolving " + str(len(wanted)) + " files on Commons ...")
    meta = fetch_metadata(sorted(set(wanted.values())))

    bad = {t: m["licence"] for t, m in meta.items() if m["licence"] not in ALLOWED_LICENCES}
    if bad:
        for title, lic in bad.items():
            print("  DISALLOWED " + repr(lic) + ": " + title, file=sys.stderr)
        return 1

    SVG_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for stem, title in sorted(wanted.items()):
        info = meta[title]
        if info["mime"] != "image/svg+xml":
            raise SystemExit(title + " is " + info["mime"] + ", not SVG")
        target = SVG_DIR / (stem + ".svg")
        if target.exists():
            # Resume. CREDITS.md attributes this file to `title`, so refuse to
            # vouch for bytes that are not the ones Commons is serving.
            blob = target.read_bytes()
            if len(blob) != info["size"]:
                raise SystemExit(
                    f"{target.name}: on disk is {len(blob)} B but {title} is "
                    f"{info['size']} B; delete it and re-run"
                )
        else:
            blob = download(info["url"])
            time.sleep(THROTTLE)
        check_safe(stem, blob)
        target.write_bytes(blob)
        manifest[stem] = {
            "file": "svg/" + stem + ".svg",
            "viewBox": view_box(blob),
            "bytes": len(blob),
            "source": info["page"],
            "author": info["artist"],
            "licence": info["licence"],
            "licence_url": info["licence_url"],
        }
        print(f"  {stem:<28} {info['licence']:<14} {len(blob):>6} B")

    # Xiangqi, second pass: one download yields four files, so it cannot share the
    # loop above. The outputs are cut from the source rather than copied, which is
    # why the byte-for-byte guard does not apply and why they are marked modified.
    xq_meta = fetch_metadata(sorted(set(XIANGQI.values())))
    for piece, title in sorted(XIANGQI.items()):
        info = xq_meta[title]
        if info["licence"] not in ALLOWED_LICENCES:
            raise SystemExit(title + ": disallowed licence " + repr(info["licence"]))
        cuts = {
            pattern.format(piece=piece) + "-" + colour
            for pattern, colour in XIANGQI_CUTS.values()
        }
        if not all((SVG_DIR / (stem + ".svg")).exists() for stem in cuts):
            blob = download(info["url"])
            time.sleep(THROTTLE)
            check_safe(piece, blob)
            for stem, cut in split_xiangqi(piece, blob).items():
                (SVG_DIR / (stem + ".svg")).write_bytes(cut)
        for stem in sorted(cuts):
            cut = (SVG_DIR / (stem + ".svg")).read_bytes()
            manifest[stem] = {
                "file": "svg/" + stem + ".svg",
                "viewBox": view_box(cut),
                "bytes": len(cut),
                "source": info["page"],
                "author": info["artist"],
                "licence": info["licence"],
                "licence_url": info["licence_url"],
                "modified": "one disc cut from the four-up source sheet",
            }
            print(f"  {stem:<28} {info['licence']:<14} {len(cut):>6} B")

    # The four-up strips are what earlier runs wrote; they are not icons.
    for piece in XIANGQI:
        stale = SVG_DIR / ("xiangqi-" + piece + ".svg")
        if stale.exists():
            stale.unlink()
            print("  removed stale strip " + stale.name)

    # Authored here, not fetched. Read from disk and recorded so a rebuild keeps them.
    for stem, description in sorted(LOCAL.items()):
        target = SVG_DIR / (stem + ".svg")
        if not target.exists():
            raise SystemExit(
                f"{target.name} is drawn in this repository and is missing; restore it "
                f"from git rather than re-running the fetch"
            )
        blob = target.read_bytes()
        check_safe(stem, blob)
        manifest[stem] = {
            "file": "svg/" + stem + ".svg",
            "viewBox": view_box(blob),
            "bytes": len(blob),
            "author": LOCAL_CREDIT["author"],
            "licence": LOCAL_CREDIT["licence"],
            "licence_url": LOCAL_CREDIT["licence_url"],
            "modified": "drawn for this repository: " + description,
        }
        print(f"  {stem:<28} {LOCAL_CREDIT['licence']:<14} {len(blob):>6} B  (local)")

    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_credits(manifest)
    print("\n" + str(len(manifest)) + " files -> " + str(SVG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
