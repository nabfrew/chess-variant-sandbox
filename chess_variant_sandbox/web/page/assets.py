"""The pages' scripts and stylesheets under names that change when their bytes do (G12).

Until G12 every page carried its stylesheet in a `<style>` and named its scripts by plain
paths answered `no-store`, so every visit fetched about 1.5 MB again, and the content
policy had to allow inline style. Now each file is answered under a name with its digest
in it -- ``/static/page/core.<12 hex>.js``, ``/static/style/page.<12 hex>.css`` -- for a
year, ``immutable``: the fonts' and the vendored files' rule, a new version is a new name
and never new bytes under an old one. The pages themselves stay ``no-store``, so a page
always names the files that go with its markup, and a page drawn before a deploy never
runs a script from after it (the old name answers the old bytes from the browser's cache,
or 404 from a server that no longer holds them; never the new bytes).

Two kinds of file:

- **Files on disk** (`serve_files`): the scripts. A name is the digest of the bytes read,
  and the bytes are held beside it, so a name answers exactly the bytes it was made from.
  A file is read again when its size or modification time moves, so an edit shows on the
  next load of a running development server, whose page then names the new digest; the
  old name is a 404 from that moment.
- **Sheets** (`sheet`): stylesheets the code writes (`style.STYLE` and the utilities, the
  site's own rules, the rule page's), fixed for the life of a process.

The plain route of a file (``/static/page/core.js``) is still answered, ``no-store``, for a
tool or a test that asks by it; no page names it.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Mapping
from pathlib import Path

# How many hex digits of the SHA-256 a name carries: the site picture's length.
DIGEST_LENGTH = 12

# Where the sheets are answered.
STYLE_PREFIX = "/static/style/"

TYPES = {".js": "text/javascript", ".css": "text/css; charset=utf-8"}

_VERSIONED = re.compile(
    rf"^(?P<stem>/.+)\.(?P<digest>[0-9a-f]{{{DIGEST_LENGTH}}})(?P<ext>\.(?:js|css))$"
)

# Plain route -> file on disk, for every script a page names.
_FILES: dict[str, Path] = {}
# Plain route -> (mtime_ns, size, digest, bytes), the last read of each file.
_HELD: dict[str, tuple[int, int, str, bytes]] = {}
# Plain route -> (digest, bytes), for the sheets.
_SHEETS: dict[str, tuple[str, bytes]] = {}
_LOCK = threading.Lock()


def digest(body: bytes) -> str:
    """The version a name carries: the first `DIGEST_LENGTH` hex digits of its SHA-256."""
    return hashlib.sha256(body).hexdigest()[:DIGEST_LENGTH]


def versioned(route: str, version: str) -> str:
    """`route` with `version` before its extension: `/a/b.js` -> `/a/b.<version>.js`."""
    stem, dot, ext = route.rpartition(".")
    return f"{stem}.{version}.{ext}" if dot else f"{route}.{version}"


def plain(path: str) -> str:
    """The plain route a versioned name stands for, or `path` itself when it is not one."""
    found = _VERSIONED.match(path)
    return found["stem"] + found["ext"] if found else path


def serve_files(table: Mapping[str, Path]) -> None:
    """Answer each plain route of `table` (route -> file) under its versioned name too."""
    with _LOCK:
        _FILES.update(table)


def _read(route: str) -> tuple[str, bytes]:
    """The digest and bytes of a file's current contents, read again only when it moved."""
    path = _FILES[route]
    stat = path.stat()
    held = _HELD.get(route)
    if held is not None and held[0] == stat.st_mtime_ns and held[1] == stat.st_size:
        return held[2], held[3]
    body = path.read_bytes()
    version = digest(body)
    _HELD[route] = (stat.st_mtime_ns, stat.st_size, version, body)
    return version, body


def sheet(name: str, text: str) -> str:
    """Register a stylesheet the code writes; its versioned route."""
    route = f"{STYLE_PREFIX}{name}.css"
    body = text.encode("utf-8")
    with _LOCK:
        _SHEETS[route] = (digest(body), body)
    return name_of(route)


def name_of(route: str) -> str:
    """The versioned name a page writes for a plain route of a file or a sheet."""
    if route in _SHEETS:
        return versioned(route, _SHEETS[route][0])
    return versioned(route, _read(route)[0])


def answer(path: str) -> tuple[bytes, str] | None:
    """What a versioned name answers -- its bytes and type -- or `None` when it names no
    file or sheet, or a version that is not the current one."""
    found = _VERSIONED.match(path)
    if found is None:
        return None
    route = found["stem"] + found["ext"]
    if route in _SHEETS:
        version, body = _SHEETS[route]
    elif route in _FILES:
        try:
            version, body = _read(route)
        except OSError:
            return None
    else:
        return None
    if version != found["digest"]:
        return None
    return body, TYPES[found["ext"]]


def sheet_text(route: str) -> str:
    """A registered sheet's text, by its plain or versioned route (for the tests)."""
    return _SHEETS[plain(route)][1].decode("utf-8")
