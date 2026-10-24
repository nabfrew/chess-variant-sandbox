"""Rebuild assets/pieces/ from Wikimedia Commons.

The library is committed, so this script is provenance rather than a build step:
it records exactly which Commons file every icon came from, re-checks the licence
of each one against an allowlist, and regenerates CREDITS.md and manifest.json.

Run from anywhere:  python assets/pieces/fetch.py
Without the network:  python assets/pieces/fetch.py --local   (re-records only the files
drawn in this repository, keeping every downloaded row as the manifest has it)
"""

# compute-lease: light
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
    # Not "c": Commons' chancellor is a knight on a rook's battlements turned
    # upside down, and a first-time visitor read it as a knight (first-visit study
    # N2). Not "a" either: Commons' archbishop is a whole knight on a bishop's foot,
    # the same mistake. Both are drawn here instead; see DERIVED.
    "amazon": "A",
    "centaur": "C",
    # short-range steppers and leapers
    "ferz": "F",
    "dabbaba": "v",
    "man": "M",
    "commoner": "x",
    # long leapers. The nightrider, zebra and unicorn ("K", "Z", "U") and the dragon
    # ("D") were Commons files once: each is a whole knight with a mark on it, which the
    # silhouette sort could not tell from the knight at 24 px (plan-visual.md §7), so they
    # are drawn here now, the knight's head on the grammar's bodies (`draw.py`).
    "giraffe": "G",
    # the rest of the fairy vocabulary
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

# Icons drawn for this repository *from* Cburnett's pieces, so they are his work
# rearranged rather than originals: credited to him, under his licence (CC BY-SA 3.0
# asks a derivative to keep it), with the change named in the notes column. Like
# LOCAL, `main()` reads them from disk and never downloads or overwrites them.
#
# The chancellor (rook + knight) is the rook's tower, shortened, with the knight's
# head rising from its battlements, so its outline reads tower first. Commons' own
# compound puts a whole knight on a sliver of upturned battlements, and at a card's
# size the head is all that reads (first-visit study N2).
#
# The archbishop (bishop + knight), by the same rule, is the bishop at nine tenths
# without the ball on its mitre, with the knight's head (at 0.6, as the chancellor's)
# rising from behind the mitre's point, so its outline reads mitre first. Commons' own
# is a whole knight standing on a bishop's foot.
_BY_CBURNETT = {
    "author": "Cburnett",
    "licence": "CC BY-SA 3.0",
    "licence_url": "http://creativecommons.org/licenses/by-sa/3.0/",
}
DERIVED = {
    **{
        "chancellor-" + colour: {
            "source": "https://commons.wikimedia.org/wiki/File:Chess_n" + c + "t45.svg",
            **_BY_CBURNETT,
            "modified": (
                "drawn for this repository from Cburnett's knight and rook "
                "(Chess_r" + c + "t45.svg): the rook's tower with the knight's head on "
                "its battlements"
            ),
        }
        for colour, c in (("white", "l"), ("black", "d"))
    },
    **{
        "archbishop-" + colour: {
            "source": "https://commons.wikimedia.org/wiki/File:Chess_n" + c + "t45.svg",
            **_BY_CBURNETT,
            "modified": (
                "drawn for this repository from Cburnett's knight and bishop "
                "(Chess_b" + c + "t45.svg): the bishop's mitre with the knight's head "
                "rising from behind its point"
            ),
        }
        for colour, c in (("white", "l"), ("black", "d"))
    },
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


# The pieces drawn here in Cburnett's grammar (plan-visual.md §7, V9): `draw.py` writes
# them, and this records each with its licence, read from what it is made of. A drawing
# of the grammar's own parts is original work released CC0; one that uses Cburnett's
# knight head is his work rearranged, credited to him under CC BY-SA 3.0. Like LOCAL and
# DERIVED, `main()` never downloads or overwrites them.
#
# The alfil, camel and wazir were Commons files once, Quivira's font glyphs (White/Black
# alfil.svg, White/Black chess camel.svg, White/Black wazir.svg), a second style beside
# Cburnett's whose white forms had to be filled to stop the square showing through. They
# are drawn here now, in the grammar (Q557b).
def drawn() -> dict[str, dict]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from draw import DRAWN, SEATS

    out: dict[str, dict] = {}
    for slug, drawing in sorted(DRAWN.items()):
        for seat in SEATS:
            if drawing.cburnett:
                letter = "l" if seat == "white" else "d"
                credit = {
                    "source": "https://commons.wikimedia.org/wiki/File:Chess_n" + letter
                    + "t45.svg",
                    **_BY_CBURNETT,
                    "modified": "drawn for this repository from Cburnett's knight: "
                    + drawing.says,
                }
            else:
                credit = {
                    **LOCAL_CREDIT,
                    "modified": "drawn for this repository in Cburnett's grammar: "
                    + drawing.says,
                }
            out[slug + "-" + seat] = credit
    return out


def record_local(manifest: dict[str, dict]) -> None:
    """Every file drawn in this repository (LOCAL, DERIVED and the grammar's), read from
    disk into `manifest`. None of them is downloaded, so this is all a rebuild without the
    network (`--local`) has to do."""
    for stem, description in sorted(LOCAL.items()):
        credit = {**LOCAL_CREDIT, "modified": "drawn for this repository: " + description}
        manifest[stem] = _from_disk(stem, credit, "local")
    for stem, credit in sorted(DERIVED.items()):
        manifest[stem] = _from_disk(stem, credit, "derived")
    for stem, credit in drawn().items():
        manifest[stem] = _from_disk(stem, credit, "drawn")


def _from_disk(stem: str, credit: dict, kind: str) -> dict:
    target = SVG_DIR / (stem + ".svg")
    if not target.exists():
        raise SystemExit(
            f"{target.name} is drawn in this repository and is missing; restore it from git "
            f"(or run draw.py) rather than re-running the fetch"
        )
    blob = target.read_bytes()
    check_safe(stem, blob)
    print(f"  {stem:<28} {credit['licence']:<14} {len(blob):>6} B  ({kind})")
    return {"file": "svg/" + stem + ".svg", "viewBox": view_box(blob), "bytes": len(blob),
            **credit}


def write_manifest(manifest: dict[str, dict]) -> None:
    (ROOT / "manifest.json").write_text(
        json.dumps(dict(sorted(manifest.items())), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def local_only() -> int:
    """`--local`: re-record the files drawn here and keep every downloaded row as it is.

    The downloaded rows are the existing manifest's, so this needs no network. A row whose
    file is gone is dropped, which is how a Commons file replaced by a drawing leaves the
    manifest and the credits.
    """
    old = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    mine = set(LOCAL) | set(DERIVED) | set(drawn())
    manifest = {
        stem: row for stem, row in old.items()
        if stem not in mine and (ROOT / row["file"]).exists()
    }
    record_local(manifest)
    write_manifest(manifest)
    write_credits(manifest)
    print("\n" + str(len(manifest)) + " files -> " + str(SVG_DIR))
    return 0


def write_credits(manifest: dict[str, dict]) -> None:
    lines = [
        "# Credits",
        "",
        "Every file in `svg/` is under the licence named below. Most came from",
        "Wikimedia Commons and are redistributed byte-for-byte unmodified; the",
        "xiangqi files are each one disc cut out of a four-up source sheet, the two",
        "chancellor and two archbishop files were drawn for this repository from",
        "Cburnett's rook, bishop and knight and keep his licence, and the six",
        "rock/paper/scissors files were drawn for this repository and released CC0.",
        "The pieces drawn in Cburnett's grammar",
        "(`draw.py`) are original work released CC0, except those built on his",
        "knight's head, which keep his CC BY-SA 3.0 and credit him.",
        "The notes column says which is which. Generated by `fetch.py`; edit that",
        "script, not this file.",
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
    # LF on every machine: a CRLF copy on disk moves the card answers' key
    # (`library.card_answers_key` hashes assets/pieces/) away from the one HEAD names.
    (ROOT / "CREDITS.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> int:
    wanted: dict[str, str] = {}  # output stem -> Commons title
    for slug, letter in CBURNETT.items():
        wanted[slug + "-white"] = "File:Chess " + letter + "lt45.svg"
        wanted[slug + "-black"] = "File:Chess " + letter + "dt45.svg"
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

    # Drawn here, not fetched. Read from disk and recorded so a rebuild keeps them.
    record_local(manifest)

    write_manifest(manifest)
    write_credits(manifest)
    print("\n" + str(len(manifest)) + " files -> " + str(SVG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(local_only() if "--local" in sys.argv[1:] else main())
