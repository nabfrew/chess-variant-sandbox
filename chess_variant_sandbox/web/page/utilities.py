"""The rules that were `style=` attributes, as classes (G12, `docs/plan-growth.md`).

The page's scripts wrote about five hundred `style="..."` attributes into their markup,
nearly all of them a margin or a gap. A content policy that refuses inline style
(`style-src 'self'`, `web/server.py`) makes every one of them inert, so each declaration is
a class here instead: `u-` then the property's short name then its value, `margin-top: 6px`
being `u-mt-6` and `margin: 6px 0 0 18px` `u-m-6-0-0-18`. An element carries one class per
declaration it had.

Each rule is `!important`, because that is how an attribute stood: above every selector in
the stylesheet, whatever its specificity, so a class keeps the place in the cascade the
attribute had. The page's few `!important` rules were already above an attribute, and still
are, as the later or more specific of two important rules. A value the script works out
(a bar's width, a card's sheet of paper) is not here: `styleClass` in `viz/wrapped.js` makes
a rule for it through the CSSOM, which the policy does not govern.

`tests/test_style_attributes.py` holds the table to the markup: every `u-` class a script or
a drawn page writes is defined, every one defined is written, and no served markup carries a
`style=` attribute.
"""

from __future__ import annotations

UTILITIES: dict[str, str] = {
    "u-ai-baseline": "align-items: baseline",
    "u-ai-flex-end": "align-items: flex-end",
    "u-ai-flex-start": "align-items: flex-start",
    "u-as-center": "align-self: center",
    "u-bc-act": "border-color: var(--act)",
    "u-bc-good": "border-color: var(--good)",
    "u-bc-zone": "border-color: var(--zone)",
    "u-bd-0": "border: 0",
    "u-bg-none": "background: none",
    "u-bt-1-solid-line": "border-top: 1px solid var(--line)",
    "u-c-act": "color: var(--act)",
    "u-c-bad": "color: var(--bad)",
    "u-c-dot": "color: var(--dot)",
    "u-c-good": "color: var(--good)",
    "u-c-ink": "color: var(--ink)",
    "u-c-zone": "color: var(--zone)",
    "u-c-muted": "color: var(--muted)",
    "u-cur-default": "cursor: default",
    "u-cur-pointer": "cursor: pointer",
    "u-d-block": "display: block",
    "u-d-contents": "display: contents",
    "u-d-flex": "display: flex",
    "u-d-inline-block": "display: inline-block",
    "u-ff-mono": "font-family: ui-monospace, monospace",
    "u-flex-1": "flex: 1",
    "u-flex-1-1-16rem": "flex: 1 1 16rem",
    "u-flex-none": "flex: none",
    "u-fs-11": "font-size: 11px",
    "u-fs-12": "font-size: 12px",
    "u-fs-14": "font-size: 14px",
    "u-fs-1_5em": "font-size: 1.5em",
    "u-fs-25": "font-size: 25px",
    "u-fs-26": "font-size: 26px",
    "u-fw-400": "font-weight: 400",
    "u-fw-600": "font-weight: 600",
    "u-gap-0": "gap: 0",
    "u-gap-3": "gap: 3px",
    "u-gap-4": "gap: 4px",
    "u-gap-4-14": "gap: 4px 14px",
    "u-gap-6": "gap: 6px",
    "u-gap-7": "gap: 7px",
    "u-gap-8": "gap: 8px",
    "u-gcol-1-n1": "grid-column: 1/-1",
    "u-h-18": "height: 18px",
    "u-h-auto": "height: auto",
    "u-jc-space-between": "justify-content: space-between",
    "u-lh-1": "line-height: 1",
    "u-lh-1_1": "line-height: 1.1",
    "u-ls-none": "list-style: none",
    "u-m-0": "margin: 0",
    "u-m-0-0-12": "margin: 0 0 12px",
    "u-m-0-0-4": "margin: 0 0 4px",
    "u-m-0-0-6": "margin: 0 0 6px",
    "u-m-0-0-8": "margin: 0 0 8px",
    "u-m-10-0-0": "margin: 10px 0 0",
    "u-m-10-0-4": "margin: 10px 0 4px",
    "u-m-12-0-4": "margin: 12px 0 4px",
    "u-m-14-0-4": "margin: 14px 0 4px",
    "u-m-4-0": "margin: 4px 0",
    "u-m-4-0-0": "margin: 4px 0 0",
    "u-m-4-0-0-18": "margin: 4px 0 0 18px",
    "u-m-4-0-6": "margin: 4px 0 6px",
    "u-m-6-0": "margin: 6px 0",
    "u-m-6-0-0-18": "margin: 6px 0 0 18px",
    "u-m-6-0-8": "margin: 6px 0 8px",
    "u-m-8-0": "margin: 8px 0",
    "u-m-8-0-0": "margin: 8px 0 0",
    "u-m-8-0-10": "margin: 8px 0 10px",
    "u-m-8-0-4": "margin: 8px 0 4px",
    "u-m-n6-0-10": "margin: -6px 0 10px",
    "u-maxw-100p": "max-width: 100%",
    "u-maxw-190": "max-width: 190px",
    "u-maxw-34rem": "max-width: 34rem",
    "u-maxw-70ch": "max-width: 70ch",
    "u-maxw-74ch": "max-width: 74ch",
    "u-mb-10": "margin-bottom: 10px",
    "u-mb-12": "margin-bottom: 12px",
    "u-mb-14": "margin-bottom: 14px",
    "u-mb-2": "margin-bottom: 2px",
    "u-mb-3": "margin-bottom: 3px",
    "u-mb-4": "margin-bottom: 4px",
    "u-mb-6": "margin-bottom: 6px",
    "u-mb-7": "margin-bottom: 7px",
    "u-mb-8": "margin-bottom: 8px",
    "u-minh-40": "min-height: 40px",
    "u-minh-44": "min-height: 44px",
    "u-minh-52": "min-height: 52px",
    "u-minw-110": "min-width: 110px",
    "u-minw-12em": "min-width: 12em",
    "u-minw-150": "min-width: 150px",
    "u-minw-170": "min-width: 170px",
    "u-minw-2_5em": "min-width: 2.5em",
    "u-minw-40": "min-width: 40px",
    "u-ml-1_4em": "margin-left: 1.4em",
    "u-mt-0": "margin-top: 0",
    "u-mt-10": "margin-top: 10px",
    "u-mt-12": "margin-top: 12px",
    "u-mt-14": "margin-top: 14px",
    "u-mt-2": "margin-top: 2px",
    "u-mt-3": "margin-top: 3px",
    "u-mt-4": "margin-top: 4px",
    "u-mt-5": "margin-top: 5px",
    "u-mt-6": "margin-top: 6px",
    "u-mt-7": "margin-top: 7px",
    "u-mt-8": "margin-top: 8px",
    "u-mt-9": "margin-top: 9px",
    "u-op-0_45": "opacity: .45",
    "u-p-0": "padding: 0",
    "u-p-18": "padding: 18px",
    "u-p-4": "padding: 4px",
    "u-p-4-8": "padding: 4px 8px",
    "u-p-5-8": "padding: 5px 8px",
    "u-pl-0": "padding-left: 0",
    "u-pl-16": "padding-left: 16px",
    "u-pt-12": "padding-top: 12px",
    "u-pt-9": "padding-top: 9px",
    "u-ta-right": "text-align: right",
    "u-td-underline": "text-decoration: underline",
    "u-td-underline-dotted": "text-decoration: underline dotted",
    "u-tf-rot180": "transform: rotate(180deg)",
    "u-va-n4": "vertical-align: -4px",
    "u-vis-hidden": "visibility: hidden",
    "u-w-100p": "width: 100%",
    "u-w-1_4em": "width: 1.4em",
    "u-w-110": "width: 110px",
    "u-w-18": "width: 18px",
    "u-w-64": "width: 64px",
    "u-w-72": "width: 72px",
    "u-w-74": "width: 74px",
    "u-w-88": "width: 88px",
    "u-wrap-nowrap": "flex-wrap: nowrap",
    "u-ws-nowrap": "white-space: nowrap",
}
"""Class -> the one declaration it stands for."""

# Cytoscape writes `.__________cytoscape_container { position: relative; }` into a `<style>`
# of its own, first in the head, unless an element with its stylesheet's id is already
# there; the page's stylesheet link carries that id (`web.page.CYTOSCAPE_SHEET_ID`) and
# this is the rule. It opens the sheet, where Cytoscape put it, so the page's own rules
# (`.exploreCanvas { position: absolute }`) still come after it and win.
VENDOR = "  .__________cytoscape_container { position: relative; }\n"

CSS = (
    "\n  /* G12: the rules that were `style=` attributes (`web/page/utilities.py`). */\n"
    + "".join(
        f"  .{name} {{ {declaration} !important; }}\n"
        for name, declaration in UTILITIES.items()
    )
)
"""The stylesheet's part, appended to `style.STYLE` in `web.page.STYLESHEET`, which
`VENDOR` opens."""
