"""The stylesheet: palette, type, panels, board, and the disclosure idiom.

One string, appended straight into the page's <style> block. Split out of the
page itself because the page is one long string literal and a stylesheet inside
it is the half nobody can find.

Three parts, in this order: `TOKENS`, every colour by what it means, light and dark
(`tokens.py`); the typefaces (`shell.faces()`); and `RULES`, which name no colour at all,
only tokens (`tests/test_style.py`). The look is plan-visual.md's club room (§3, §9).
"""

from __future__ import annotations

from chess_variant_sandbox.web.landing import BLOCKS
from chess_variant_sandbox.web.page.shell import BODY, HEAD, MONO, faces
from chess_variant_sandbox.web.page.tokens import TOKENS

RULES = """
  * { box-sizing: border-box; }
  /* The ground is paper: the fibre tile over the ground colour (tokens.py's `paper`, a few
     hundred bytes of inline SVG noise, too faint to move a contrast ratio below its floor;
     `tests/test_style.py` reads the worst fleck). */
  body {
    margin: 0; padding: 0 20px 48px; color: var(--ink);
    background: var(--paper) 0 0 / 180px, var(--bg);
    font: 16px/1.5 var(--f-body);
  }
  /* --- the header (plan-visual.md §3.4, V2): one line of 56 px on both servers. The mark
     and the name; the tagline, where a view writes its one-line summary; the sections; the
     corner. On a phone the tagline goes and the sections fold behind the menu button
     (`shell.js`); a page whose script did not run wraps them under the name instead. --- */
  header.top {
    position: relative; display: flex; align-items: center; flex-wrap: wrap; gap: 6px 18px;
    min-height: 56px; padding: 6px 0; border-bottom: 1px solid var(--line); margin-bottom: 22px;
  }
  header.top > * { flex: none; }
  header.top .grow { flex: 1 1 0; }
  .brand { margin: 0; font-size: 19px; line-height: 1; color: var(--ink); cursor: pointer; }
  .brand, .brand a { display: flex; align-items: center; gap: 10px; }
  header .brand a { color: var(--ink); border-bottom: none; }
  .brand .mark { flex: none; display: block; }
  header.top .tagline { flex: 0 1 auto; min-width: 0; font-size: 15px; white-space: nowrap;
                        overflow: hidden; text-overflow: ellipsis; }
  .sections { display: flex; gap: 18px; font-size: 15px; }
  header .sections a { color: var(--ink); opacity: .72; padding: 4px 0;
                       border-bottom: 2px solid transparent; }
  header .sections a:hover { opacity: 1; color: var(--ink); }
  header .sections a[aria-current] { opacity: 1; border-bottom-color: var(--act); }
  header.top #runs:empty, header.top #who:empty { display: none; }
  .menu { display: none; }
  @media (max-width: 760px) {
    header.top { gap: 6px 12px; }
    header.top .tagline { display: none; }
    .brand { font-size: 16px; }
    .brand, .brand a { gap: 8px; }
    /* The corner takes what the name and the menu leave and no more: a long handle, or the
       maintainer's two links, end in an ellipsis rather than pushing the page sideways. */
    header.top #who, header.top #runs { flex: 0 1 auto; min-width: 0; overflow: hidden;
                                        text-overflow: ellipsis; white-space: nowrap; }
    /* Cut from the left, so what stays is the corner's last word: the person's own name. */
    header.top #who { direction: rtl; }
    /* A visitor's corner is shortened on purpose to *sign in* (LP0): cut from the left it
       read *… count*, the one word that says nothing without the rest. The reason is on
       the sign-in page and the landing page in full. */
    header.top #who .accountWhy { display: none; }
    .js header.top { flex-wrap: nowrap; }
    .js .menu { display: inline-flex; align-items: center; justify-content: center; order: 10;
                width: var(--ctl-h); height: var(--ctl-h); padding: 0; color: var(--ink); }
    .js .sections { display: none; }
    .js.menu-open .sections {
      display: flex; flex-direction: column; gap: 0; position: absolute; z-index: 50;
      right: 0; top: calc(100% + 4px); min-width: 200px; padding: 4px 16px;
      background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
      box-shadow: var(--shadow-pop);
    }
    .js.menu-open header .sections a { padding: 11px 0; border-bottom: 1px solid var(--line);
                                       opacity: 1; }
    .js.menu-open header .sections a:last-child { border-bottom: none; }
    .js.menu-open header .sections a[aria-current] { color: var(--act); }
  }
  /* The type scale (plan-visual.md §3.1), size/line height in px: 28/34 a page's title,
     20/26 a section, 16/24 body, 13/18 small, 14/20 mono. Headings in the serif. */
  h1, h2, h3, h4 { font-family: var(--f-head); font-weight: 600; letter-spacing: -.01em; }
  h1 { font-size: 19px; margin: 0; cursor: pointer; }
  h2 { font-size: 20px; line-height: 26px; margin: 0 0 8px; }
  h3 { font-size: 16px; line-height: 22px; margin: 18px 0 6px; color: var(--ink); }
  h4 { font-size: 15px; }
  code, kbd, samp { font-family: var(--f-mono); font-size: .92em; }
  a { color: inherit; }
  header a, footer a { color: var(--muted); text-decoration: none;
                       border-bottom: 1px solid var(--line); }
  header a:hover, footer a:hover { color: var(--ink); }
  footer { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 18px;
           padding: 22px 0 6px; margin-top: 30px; border-top: 1px solid var(--line); }
  footer #foot { flex: 1 1 auto; }
  /* A member's mark (R2, archive/plan-return.md §6.3): their own invader, inline beside their name
     wherever the name is shown. In the header's corner only at a desk: on a phone the
     corner is cut to the handle's last letters. */
  .memberMark { display: inline-block; flex: none; vertical-align: -4px; margin-right: 6px; }
  h1 .memberMark { vertical-align: -5px; margin-right: 12px; }
  @media (max-width: 760px) { header.top #who .memberMark { display: none; } }
  /* The way to your own page (Q731): a button, not a name. A bare handle in the corner
     read as a label saying who is signed in; this is a sheet of the panel's paper with the
     controls' outline, height and radius, the handle, then *your page* and a chevron in the
     action colour. On a phone the handle goes and *your page* stays (`site/pages.py`'s
     `_member_button`, `core.js`'s `memberButton`). */
  header a.memberButton {
    display: inline-flex; align-items: center; gap: 8px; direction: ltr; white-space: nowrap;
    min-height: var(--ctl-h); padding: 0 12px; margin-left: 4px; vertical-align: middle;
    border: 1px solid var(--edge); border-radius: var(--r-ctl);
    background: var(--paper) 0 0 / 180px, var(--panel); box-shadow: var(--shadow);
    color: var(--ink); font: 500 15px/1 var(--f-body);
  }
  header a.memberButton:hover { border-color: var(--ink); color: var(--ink); }
  header a.memberButton .memberMark { margin-right: 0; vertical-align: 0; }
  header a.memberButton .memberGo { color: var(--act); font-weight: 600; }
  header a.memberButton .memberName + .memberGo { padding-left: 8px;
                                                  border-left: 1px solid var(--line); }
  /* On a phone the corner has about 50 px beside the name and the menu, too little for a
     button that says where it goes. So when the corner holds the button, the button keeps
     its whole width and the site's name gives way instead, onto two lines; the
     maintainer's *Maintenance* link and its dot leave the corner (their own page has it). */
  @media (max-width: 760px) {
    header a.memberButton { font-size: 14px; padding: 0 10px; }
    header a.memberButton .memberName { display: none; }
    header a.memberButton .memberName + .memberGo { padding-left: 0; border-left: none; }
    header.top:has(.memberButton) .brand { flex: 0 1 auto; min-width: 0; line-height: 1.1; }
    header.top #who:has(.memberButton) { flex: none; overflow: visible; direction: ltr;
                                         font-size: 0; }
    header.top #who:has(.memberButton) > a:not(.memberButton) { display: none; }
  }
  /* The footer is signed (V7): the mark small and the name in the heading face, then the
     links, then the theme switch; on a phone each takes its own line as it wraps. */
  footer .sign { display: inline-flex; align-items: center; gap: 8px; color: var(--muted);
                 font: 600 14px/20px var(--f-head); }
  footer .sign .mark { display: block; flex: none; }
  /* The theme switch (§3.6): light, dark, or the system's, kept in this browser. */
  .wrap { max-width: 1120px; margin: 0 auto; }
  .muted { color: var(--muted); }
  .small { font-size: 13px; line-height: 18px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); }
  /* A panel is a sheet of the same paper, lifted: the fibre over the panel colour, a soft
     shadow under it (the club room's `--shadow`; none in dark, where a shadow is mud). */
  .panel { background: var(--paper) 0 0 / 180px, var(--panel); border: 1px solid var(--line);
           border-radius: 10px; padding: 15px; box-shadow: var(--shadow); }
  .card { cursor: pointer; transition: border-color .18s ease, transform .18s ease; }
  /* Each library card is its own sheet (Q684): `wood.js`'s `paperSheet` sets where in the
     fibre tile the sheet is cut, its size, its turn and its tone from the game's hash. The
     fibre moves to a layer under the card's text, a little larger than the card so a turned
     sheet still covers it, and the card clips it; the sheet's tone is the panel's lightness
     and chroma moved together (`tokens.SHEET`), where the browser can compute it, and the
     plain panel where it cannot. Nothing here changes the card's box. */
  .panel.card { position: relative; isolation: isolate; overflow: clip;
                background: var(--panel);
                background: oklch(from var(--panel) calc(l + var(--sheet-t, 0) * .006)
                                  calc(c - var(--sheet-t, 0) * .002) h); }
  .panel.card::before { content: ""; position: absolute; inset: -32px; z-index: -1;
                        pointer-events: none; transform: rotate(var(--sheet-a, 0deg));
                        background: var(--paper) var(--sheet-x, 0) var(--sheet-y, 0) /
                                    calc(180px * var(--sheet-s, 1)); }
  .card:hover { border-color: var(--muted); transform: translateY(-1px); }
  /* The card's layout (V6, plan-visual.md §3.4 and S5): the opening position at 104 px,
     the words beside it; three cards a row at 1120 px, one a row on a phone with an 80 px
     board. A card whose game drew no board is the words alone. */
  .card:has(> .thumb) { display: grid; grid-template-columns: 104px minmax(0, 1fr);
                        gap: 14px; align-items: start; }
  .card .thumb { pointer-events: none; }
  /* One loud thing: the level and the score are the pills; the rest of what the checks
     found is a quiet line of the same words, each with its hover, no border. */
  .card .notes { gap: 4px 10px; margin-bottom: 4px; }
  .card .notes .tag { border: 0; padding: 0; border-radius: 0; font-size: 12px; }
  .card .notes .tag.score { color: var(--good); font-weight: 600; }
  @media (max-width: 760px) {
    .grid { grid-template-columns: minmax(0, 1fr); gap: 10px; }
    .card:has(> .thumb) { grid-template-columns: 80px minmax(0, 1fr); gap: 12px; }
    /* A phone's card is shorter: the kind of game is on the game's page and in the
       filters, and two lines of the description say what it is. */
    .grid > .card { padding: 12px; }
    .card .kind { display: none; }
    .card p { -webkit-line-clamp: 2; }
  }
  /* The landing page (Q718): the words beside a quiet board at 1120 px, above it on a
     phone; then the sections as one column of reading width. The board's pieces are set
     down by fading in (opacity only: a piece's own turn is its transform), and a game gives
     way to the next by fading out; with reduced motion neither happens (`landing.js` draws
     one still board, and this rule shows any piece already marked). */
  .landingHero { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 380px);
                 gap: 28px; align-items: center; margin: 8px 0 30px; }
  .landingPart, .landingFoot { max-width: 760px; }
  .landingTitle { font-size: 28px; line-height: 34px; margin: 0 0 12px; }
  .landingLead { font-size: 18px; line-height: 26px; margin: 0 0 18px; }
  .landingWays { margin-bottom: 10px; }
  .landing a.button { display: inline-flex; align-items: center; min-height: var(--ctl-h);
                      padding: 7px 14px; border: 1px solid var(--line); border-radius: var(--r-ctl);
                      background: var(--panel); color: var(--ink); text-decoration: none;
                      font-weight: 500; }
  .landing a.button:hover { border-color: var(--muted); }
  .landing a.button.primary { background: var(--act); border-color: var(--act);
                              color: var(--act-ink); font-weight: 600; }
  .landingShow { margin: 0; }
  .landingBoard { aspect-ratio: 1; display: flex; align-items: center; justify-content: center; }
  .landingSlide { transition: opacity .5s ease; }
  .landingSlide.leaving { opacity: 0; }
  .landingSlide .thumb { width: 100%; }
  .landingBoard .settle { opacity: 0; animation: landingSettle .6s ease-out forwards; }
  @keyframes landingSettle { to { opacity: 1; } }
  @media (prefers-reduced-motion: reduce) {
    .landingSlide { transition: none; }
    .landingBoard .settle { opacity: 1; animation: none; }
  }
  /* Three lines of caption and the controls' row are held from the start (Q741): the
     longest sentence of the starting games takes three lines on a phone, and the
     controls arrive with the listing, so neither moves *Made by members* under it. */
  .landingShow figcaption { margin-top: 10px; min-height: 54px; }
  .landingShowControls { gap: 4px; margin-top: 2px; min-height: var(--ctl-h); }
  .landingPart { margin: 0 0 26px; }
  .landingPart h3 { font-size: 20px; line-height: 26px; margin: 0 0 8px; }
  .landingPart p { margin: 0 0 10px; }
  .landingFeatures { margin: 0; padding-left: 20px; }
  .landingFeatures li { margin-bottom: 6px; }
  .landingMore { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 10px; }
  .landingMore > summary { min-height: var(--hit); display: flex; align-items: center;
                           font-weight: 600; }
  .landingMoreBody { padding-top: 8px; }
  .landingMore .quality { margin: 18px 0 0; }
  .landingMore h4 { font-size: 16px; line-height: 22px; margin: 0 0 2px; }
  .landingMore .quality > p { margin: 0 0 6px; }
  .ingredients, .ruleOuts { margin: 0; padding: 0; list-style: none; }
  .ruleOuts { padding-left: 20px; list-style: disc; }
  .ruleOuts li { margin-bottom: 4px; }
  .ingredient { border-left: 3px solid var(--line); padding: 2px 0 2px 12px; margin: 10px 0; }
  .ingredient p { margin: 2px 0 4px; font-size: 14px; line-height: 20px; }
  .ingredientHead { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .ingredientWeight { font-family: var(--f-mono); font-size: 13px; color: var(--muted); }
  .landingFoot:empty { display: none; }
  /* LP2: the palette as a wall of words (`landing.BLOCKS`), one row a family: its name with
     the family's colour (`--family`, set on the row by the hue rules after these) beside its
     words, each a link to its section of /rules. Wider than the reading column, so a row's
     words sit on a line or two at 1400; on a phone each row's words are one rail that
     scrolls sideways inside the row, and the page itself never does. */
  .landingWall { max-width: none; }
  .landingWall > p, .landingWall > h3 { max-width: 760px; }
  .wall { margin: 14px 0 0; display: grid; }
  .wallRow { display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 16px;
             align-items: baseline; padding: 12px 0; border-top: 1px solid var(--line);
             --family: var(--muted); }
  .wallRow dt { font-family: var(--f-head); font-size: 18px; line-height: 24px;
                display: flex; gap: 8px; align-items: center; }
  .wallRow dd { margin: 0; display: flex; flex-wrap: wrap; gap: 8px; min-width: 0; }
  .wall .swatch { width: 10px; height: 10px; border-radius: 3px; flex: none;
                  background: var(--family); }
  .landing a.wallWord { display: inline-flex; align-items: center; min-height: var(--chip-h);
                        padding: 4px 12px; border: 1px solid var(--line); border-radius: 999px;
                        background: var(--panel); color: var(--ink); font-size: 14px;
                        line-height: 18px; text-decoration: none; white-space: nowrap;
                        transition: border-color .2s, box-shadow .2s, transform .2s; }
  .landing a.wallWord:hover, .landing a.wallWord:focus-visible {
    border-color: var(--family); box-shadow: inset 3px 0 0 var(--family);
    transform: translateY(-1px); }
  .wallBand { margin-top: 26px; padding: 20px 24px; border-radius: 12px;
              background: var(--act-tint); display: flex; flex-wrap: wrap; gap: 12px 24px;
              align-items: center; justify-content: space-between; }
  .wallBand p { margin: 0; font-family: var(--f-head); font-size: 24px; line-height: 30px; }
  @media (prefers-reduced-motion: reduce) {
    .landing a.wallWord { transition: none; }
    .landing a.wallWord:hover, .landing a.wallWord:focus-visible { transform: none; }
  }
  @media (max-width: 760px) {
    .wallRow { grid-template-columns: minmax(0, 1fr); gap: 6px; padding: 10px 0; }
    .wallRow dd { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none;
                  overscroll-behavior-x: contain; padding: 2px 0;
                  mask-image: linear-gradient(to right, var(--ink) 82%, transparent); }
    .wallRow dd.atEnd { mask-image: none; }
    .wallBand { padding: 16px; }
    .wallBand p { font-size: 21px; line-height: 27px; }
  }
  /* Q741: *Made by members*, straight under the opening board. Wider than the reading
     column, so four games sit across at 1120 px. The shelf is one row of a fixed height
     whatever it holds -- nothing while the listing loads, up to four games, or the
     invitation before anybody has published -- so filling it moves nothing below it; on a
     phone the row scrolls sideways inside its own box, the next game showing at the edge. */
  .landingMembers { max-width: none; }
  .landingMembers > p { max-width: 760px; }
  .membersHead { display: flex; align-items: baseline; justify-content: space-between;
                 gap: 12px; }
  .membersHead a { white-space: nowrap; }
  .membersShelf { display: grid; grid-auto-flow: column; gap: 12px; height: 140px;
                  grid-auto-columns: minmax(250px, 1fr); overflow-x: auto; overflow-y: hidden;
                  scroll-snap-type: x mandatory; overscroll-behavior-x: contain;
                  scrollbar-width: thin; }
  .memberGame { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 12px;
                align-items: center; padding: 10px; min-width: 0; scroll-snap-align: start;
                box-shadow: none; }
  .memberGame .thumb { width: 92px; pointer-events: none; }
  .memberGame, .newGame { cursor: pointer; }
  .memberGameText { display: flex; flex-direction: column; align-items: flex-start;
                    gap: 4px; min-width: 0; }
  .memberGameName { font-weight: 600; line-height: 20px; max-width: 100%; overflow: hidden;
                    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .memberGameBy { display: flex; align-items: center; gap: 5px; max-width: 100%;
                  color: var(--muted); }
  .memberGameBy a { display: inline-flex; align-items: center; gap: 5px; min-width: 0; }
  .memberGameBy span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .landing a.button.memberGamePlay { min-height: var(--ctl-h-sm); padding: 4px 16px; margin-top: 2px; }
  .membersEmpty { grid-column: 1 / -1; display: flex; align-items: center; gap: 16px;
                  padding: 10px 16px; border: 1px dashed var(--line); border-radius: 10px; }
  .membersEmpty p { margin: 0; max-width: 520px; }
  .membersEmpty .button { flex: none; }
  /* Q788: *New in the library*, a shelf like *Made by members* (one row of one height,
     sideways on a phone), each card a board, the name, two lines of its sentence and the
     pills; then the picks row, the day's game, the spotlight and this month's game, side by
     side where there is room and one above the other on a phone. */
  .landingShelf, .landingPicks { max-width: none; }
  .landingShelf > p { max-width: 760px; }
  .newShelf { display: grid; grid-auto-flow: column; gap: 12px; height: 156px;
              grid-auto-columns: minmax(250px, 1fr); overflow-x: auto; overflow-y: hidden;
              scroll-snap-type: x mandatory; overscroll-behavior-x: contain;
              scrollbar-width: thin; }
  .newGame { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 12px;
             align-items: center; padding: 10px; min-width: 0; scroll-snap-align: start;
             box-shadow: none; }
  .newGame .thumb { width: 92px; pointer-events: none; }
  .newGameText { display: flex; flex-direction: column; align-items: flex-start; gap: 4px;
                 min-width: 0; }
  .newGameSentence { margin: 0; color: var(--muted); overflow: hidden; display: -webkit-box;
                     -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
  .newGamePills { gap: 5px; flex-wrap: wrap; }
  .tag.newTag { border-color: var(--act); color: var(--ink); }
  .landingPicks { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr));
                  align-items: stretch; }
  /* A picks panel is `.landingPicks > .pick`, never a bare `.pick`: `label.pick` is the
     labelled control everywhere else (the map's settings, the designer's *royal*, the
     library's filters), and a bare rule gave each a 200 px grid box (Q777). */
  .landingPicks > .pick { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 14px;
          align-items: start; margin: 0; padding: 14px; min-height: 200px; box-shadow: none; }
  .pickBoard .thumb { width: 120px; pointer-events: none; }
  /* The spotlight is the row's own width, its board larger; the other two share a row. */
  #landingSpotlight { grid-column: 1 / -1; grid-template-columns: 180px minmax(0, 1fr);
                      gap: 20px; min-height: 0; }
  #landingSpotlight .pickBoard .thumb { width: 180px; }
  #landingSpotlight .pickText { max-width: 760px; }
  .pickText h4 { font-size: 18px; line-height: 24px; margin: 2px 0 4px; }
  .pickText p { margin: 0 0 8px; }
  .pickWays { gap: 6px; flex-wrap: wrap; align-items: center; }
  .landingPicks > .pick[aria-busy] > * { visibility: hidden; }
  @media (max-width: 760px) {
    .newShelf { grid-auto-columns: minmax(270px, 86%); }
    .landingPicks { grid-template-columns: minmax(0, 1fr); }
    .landingPicks > .pick { grid-template-columns: 84px minmax(0, 1fr); gap: 12px; padding: 12px; min-height: 0; }
    .pickBoard .thumb { width: 84px; }
    #landingSpotlight { grid-template-columns: 84px minmax(0, 1fr); gap: 12px; }
    #landingSpotlight .pickBoard .thumb { width: 84px; }
  }
  /* LP3: the five shelves as one band at the top of `/game` (`landing.shelves_html`), above
     the search: the three picks side by side, each a smaller board and three lines of its
     sentence, then the two shelves; on a phone the picks are a rail too, as the shelves
     are, so the band is three rows tall and the search is near. */
  .shelves { margin: 0 0 20px; padding-bottom: 12px; border-bottom: 1px solid var(--line); }
  .shelves .landingPart { margin-bottom: 16px; }
  .shelves .landingPicks { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .shelves .landingPicks > .pick, .shelves #landingSpotlight {
    grid-column: auto; grid-template-columns: 96px minmax(0, 1fr); gap: 12px; padding: 12px;
    min-height: 0; }
  .shelves .pickBoard .thumb, .shelves #landingSpotlight .pickBoard .thumb { width: 96px; }
  .shelves .pickText h4 { font-size: 16px; line-height: 22px; }
  .shelves .pickText p { overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2;
                         -webkit-box-orient: vertical; }
  /* A band, not a page: each pick keeps its label, name, sentence and ways in; the notes
     under them (why it was picked, how this month's game works, the practice note) and the
     shelves' leads are the landing's and the game's own page's to say. */
  .shelves .pickText p.small, .shelves .landingShelf > p, .shelves .landingMembers > p {
    display: none; }
  .shelvesScore { margin: 0; }
  @media (max-width: 760px) {
    .shelves .landingPicks { grid-template-columns: none; grid-auto-flow: column;
      grid-auto-columns: minmax(270px, 86%); overflow-x: auto; overflow-y: hidden;
      scroll-snap-type: x mandatory; overscroll-behavior-x: contain; scrollbar-width: thin; }
    .shelves .landingPicks > .pick, .shelves #landingSpotlight {
      grid-template-columns: 84px minmax(0, 1fr); scroll-snap-align: start; }
    .shelves .pickBoard .thumb, .shelves #landingSpotlight .pickBoard .thumb { width: 84px; }
  }
  .scoreView .landingLead { max-width: 760px; }
  @media (max-width: 760px) {
    .landingHero { grid-template-columns: minmax(0, 1fr); gap: 18px; margin-bottom: 22px; }
    .landingShow { max-width: 340px; width: 100%; margin: 0 auto; }
    .landingTitle { font-size: 22px; line-height: 28px; }
    .landingLead { font-size: 16px; line-height: 24px; }
    .membersShelf { grid-auto-columns: minmax(250px, 82%); }
    .membersEmpty { flex-direction: column; align-items: flex-start; gap: 8px;
                    font-size: 14px; line-height: 20px; }
  }
  .card h2 { font-size: 17px; line-height: 22px; margin: 0 0 6px; }
  /* A row of their own, whatever the name is: beside the name they sat next to a short
     one and dropped under a long one, so one card was two layouts. */
  .card .pills { gap: 5px; margin-bottom: 6px; }
  .card .kind { margin-top: 5px; letter-spacing: .01em; }
  .card p { margin: 6px 0 0; font-size: 13px; color: var(--muted);
            display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .tag { font-size: 11px; padding: 2px 7px; border-radius: 20px; border: 1px solid var(--line);
         color: var(--muted); white-space: nowrap; }
  .tag.score { border-color: var(--good); color: var(--good); }
  .tag.gated { border-color: var(--bad); color: var(--bad); }
  /* Neither good nor bad: a stale measurement is a statement about the instrument
     rather than about the game, so it must not read as a verdict either way. */
  .tag.stale { border-style: dashed; }
  /* Measured on somebody's own machine and checked, not measured by the server (Q312).
     Dotted, so it never reads as the solid green of an official score. */
  .tag.provisional { border-style: dotted; color: var(--ink); }
  /* Where a game stands (Q503b): the first pill on a card, in ink, because it is the
     card's first fact rather than a verdict. */
  .tag.level { color: var(--ink); border-color: var(--muted); }
  /* The designer's four steps (`trackHtml`), one line of dots that wraps on a phone. */
  .track { display: flex; flex-wrap: wrap; gap: 4px 14px; list-style: none; padding: 0;
           margin: 4px 0 8px; font-size: 13px; color: var(--muted); }
  .track li::before { content: "\\25CB"; margin-right: 5px; }
  .track li.lit { color: var(--ink); font-weight: 600; }
  .track li.lit::before { content: "\\25CF"; }
  /* A phone keeps the four steps on one line (archive/plan-game-pipeline §3.1), which at 13px
     put *Library* on a line of its own at 375 px (Q517). */
  @media (max-width: 480px) {
    .track { flex-wrap: nowrap; gap: 4px 9px; font-size: 12px; }
    .track li { white-space: nowrap; }
    .track li::before { margin-right: 3px; }
  }
  /* A game kept on purpose for what it shows (`exhibitHtml`). Amber, the verdict's
     colour, because the sentence is neither a mark for the game nor against it: it says
     how to read the number beside it. */
  .exhibit { margin: 6px 0; padding: 3px 0 3px 9px; border-left: 3px solid var(--note);
             font-size: 12px; line-height: 1.45; color: var(--ink); }
  /* A preview (`previewHtml`, Q931): a new kind of game still being tested. The exhibit's
     amber rule, for the exhibit's reason: it says how to read everything beside it. */
  .previewNote { margin: 6px 0; padding: 3px 0 3px 9px; border-left: 3px solid var(--note);
                 font-size: 12px; line-height: 1.45; color: var(--ink); }
  .tag.level.preview { border-color: var(--note); font-weight: 600; }
  /* How far to trust a score (`trustHtml`, Q647). The exhibit's amber rule, for the
     exhibit's reason: it says how to read the number beside it. */
  .trust { margin: 8px 0; padding: 3px 0 3px 9px; border-left: 3px solid var(--note);
           font-size: 12px; line-height: 1.45; color: var(--ink); }
  .trustMore { margin-top: 4px; }
  .trustMore summary { cursor: pointer; }
  .trustMore ul { margin: 4px 0 0; }
  /* Whether people's play has beaten the strongest computer player (`opponentLineHtml`,
     Q653). Beside the trust line and never inside it, so its rule is the muted line's,
     not the trust line's amber: it reads people's games, not the score. */
  .opponentLine { margin: 10px 0 0; padding: 3px 0 3px 9px; border-left: 3px solid var(--muted);
                  font-size: 12px; line-height: 1.45; color: var(--ink); }
  /* Your best result against each computer player, weakest first (`ladderRowHtml`,
     R7): one row of pills that wraps on a phone. */
  .ladderRow { display: flex; flex-wrap: wrap; gap: 5px; list-style: none; padding: 0;
               margin: 6px 0 0; }
  .ladderRow .tag { white-space: normal; }
  .ladderRow .tag.won { border-color: var(--good); color: var(--ink); }
  .ladderRow .tag.drew { color: var(--ink); }
  /* --- the controls (plan-visual.md §3.5, V3) ---
     Four kinds of button. *Secondary* is the default, outlined. *Primary* is filled in the
     action colour, one a view, and names its result. *Quiet* is text in the action colour,
     for a board's bar and inline actions. *Danger* is outlined in the alarm red: resign, a
     revoke, a maintainer's kill (`.grave` on the site). One height scale, from the tokens:
     `--ctl-h` for a button, an input and a select; `--ctl-h-sm` for a compact control in a
     dense row; both 44 px under a coarse pointer. Disabled is 45%, never a grey of its own.
     A checkbox, a radio and a range keep the browser's drawing, in the action colour. */
  button, select, textarea, input:where(:not([type=checkbox], [type=radio], [type=range])) {
    font: inherit; color: var(--ink); background: var(--panel);
    border: 1px solid var(--line); border-radius: var(--r-ctl);
  }
  button { min-height: var(--ctl-h); padding: 7px 14px; cursor: pointer; font-weight: 500; }
  select, input:where(:not([type=checkbox], [type=radio], [type=range])) {
    min-height: var(--ctl-h); padding: 7px 12px; border-color: var(--edge);
  }
  textarea { padding: 7px 12px; border-color: var(--edge); }
  input[type=checkbox], input[type=radio], input[type=range] { accent-color: var(--act); }
  button:hover:not(:disabled) { border-color: var(--muted); }
  button:disabled { opacity: .45; cursor: default; }
  button.primary { background: var(--act); border-color: var(--act); color: var(--act-ink);
                   font-weight: 600; }
  button.primary:hover:not(:disabled) { border-color: var(--act); filter: brightness(.93); }
  button.quiet { background: transparent; border-color: transparent; color: var(--act);
                 font-weight: 600; padding-left: 10px; padding-right: 10px; }
  button.quiet:hover:not(:disabled) { border-color: transparent; text-decoration: underline;
                                       text-underline-offset: 3px; }
  button.danger, button.grave { background: transparent; border-color: var(--alarm); color: var(--alarm); }
  button.danger:hover:not(:disabled), button.grave:hover:not(:disabled) { border-color: var(--alarm); background: var(--sunk); }
  /* Keyboard focus is a ring in the action colour, never the browser's default, and never
     shown for a click. A field's own edge turns too, since a ring round a text box is
     drawn on the box. */
  :where(a, button, input, select, textarea, summary, [role=button], [tabindex]:not([tabindex="-1"])):focus-visible {
    outline: 2px solid var(--focus); outline-offset: 2px;
  }
  :where(input, select, textarea):focus-visible { outline-offset: 0; border-color: var(--focus); }
  /* A control drawn smaller than a target (the (i), a chip) reaches past its drawing to
     `--hit`: 24 px at a desk, 44 under a finger. An invisible box centred on it; a press
     on it is a press on the control. */
  button.info, .chip { position: relative; }
  button.info::after, button.chip::after {
    content: ""; position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
    width: max(100%, var(--hit)); height: max(100%, var(--hit));
  }
  /* A segmented switch, for two or three exclusive views: *list · map*, *moves · rules ·
     pieces*, the footer's *light · dark · system*. One outline round the set; the chosen
     one filled in ink, not the accent (the accent is *you can do this*, and the chosen view
     is already done). */
  .viewSwitch, .tabs, .themePick {
    display: inline-flex; max-width: 100%; border: 1px solid var(--line);
    border-radius: var(--r-ctl); overflow: hidden; background: var(--panel);
  }
  .viewSwitch button, .tabs button, .themePick button {
    min-height: max(var(--hit), calc(var(--ctl-h) - 2px)); padding: 6px 14px; border: 0; border-radius: 0;
    background: transparent; color: var(--ink);
  }
  .viewSwitch button + button, .tabs button + button, .themePick button + button {
    border-left: 1px solid var(--line);
  }
  .viewSwitch button:hover:not(.on), .tabs button:hover:not(.on),
  .themePick button[aria-pressed="false"]:hover { background: var(--sunk); }
  .viewSwitch button.on, .tabs button.on, .themePick button[aria-pressed="true"] {
    background: var(--ink); color: var(--panel); font-weight: 600;
  }
  .viewSwitch button:focus-visible, .tabs button:focus-visible,
  .themePick button:focus-visible { outline-offset: -3px; }
  /* The footer's switch is the compact size: it is a setting, not a view. */
  .themePick button { min-height: max(var(--hit), calc(var(--ctl-h-sm) - 2px)); padding: 3px 12px;
                      font-size: 13px; line-height: 18px; }
  /* A chosen option among several that are not views (a chip, a piece in hand, the
     designer's board mode, a palette entry): the action's tint, bordered in the action. */
  .chip.on, .hand button.on, .benchTabs button.on, .teachPanel button.on, .palette button.on,
  .listControl button.on, .mapLocal button.on {
    border-color: var(--act); background: var(--act-tint); color: var(--ink); font-weight: 600;
  }
  textarea { width: 100%; min-height: 74px; resize: vertical; }
  .split { display: grid; gap: 20px; grid-template-columns: minmax(320px, 1fr) minmax(280px, 380px); }
  /* A board with nothing beside it. The designer plays a draft inside the column it
     edits in, where the rules and the pieces are already on screen in the column to its
     right, so that second track would be an empty one taking width off the board. */
  .split.alone { grid-template-columns: minmax(0, 1fr); }
  @media (max-width: 860px) {
    .split { grid-template-columns: 1fr; }
    /* A grid item is at least as wide as its own content unless it is told otherwise,
       so one long piece description pushed the whole column past the screen and the
       page scrolled sideways. Stacked there is one track and it is the screen. */
    .split > * { min-width: 0; }
    /* Stacked, the board page's side panel comes a screen below the board, and its tabs
       are what a person scrolls down for (a first game opens on the rules, K3): they lead
       it, and the seat lines, which the status strip above the board already names, follow
       (Q728). */
    .boardSide { display: flex; flex-direction: column; }
    .boardSide > .tabs { margin-top: 0 !important; align-self: flex-start; }
    .boardSide > .seatLines { order: 1; margin-top: 12px; padding-top: 10px;
                              border-top: 1px solid var(--line); }
  }
  /* The game's page (V6, §3.4): the play panel and what sits beside it first, then the
     rest of the game in two columns of sections, each folding under its heading. On a
     phone it is one column in that order. Spacing inside a column is the column's gap,
     whatever margins a panel was drawn with. */
  .gameAbout { margin: 8px 0 0; max-width: 72ch; }
  .gameTop, .gameRest { display: grid; gap: 20px; align-items: start; margin-top: 16px;
                        grid-template-columns: minmax(0, 7fr) minmax(0, 5fr); }
  .gameRest { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); margin-top: 8px; }
  .gamePlay, .gameSide, .gameRest > div { display: flex; flex-direction: column; gap: 14px;
                                          min-width: 0; }
  .gamePlay > *, .gameSide > * { margin-top: 0 !important; margin-bottom: 0 !important; }
  .gamePlay > :empty, .gameSide > :empty { display: none; }
  .gameRest > div { gap: 4px; }
  @media (max-width: 860px) {
    .gameTop, .gameRest { grid-template-columns: minmax(0, 1fr); }
  }
  /* Q993b: a game's proven positions on a page of their own (docs/archive/plan-proven.md §2.2): the
     frame and the list side by side, as the game page draws its play column and its side,
     the frame first on a phone. A row is a link: its number is the one loud thing (S5),
     the side and *one winning move* its muted lines, and the reader's state a word with an
     ink tick on the right, never a colour (S4). The current row is neutral, as the move
     list's is. */
  .proven-frame .frame-status { --strip-lines: 3; }
  @media (min-width: 561px) { .proven-frame .frame-status { --strip-lines: 2; } }
  .frame-status #provenStatus { display: -webkit-box; -webkit-box-orient: vertical;
                                -webkit-line-clamp: var(--strip-lines); overflow: hidden; min-width: 0; }
  .proven-still { width: 100%; margin: 0 auto 0 0; }
  .proven-still svg { display: block; width: 100%; height: auto; }
  .proven-frame .frame-controls { margin-top: 8px; }
  .proven-list .chips { margin: 8px 0 4px; }
  .proven-group { margin-top: 8px; }
  .proven-group > summary { font-weight: 600; cursor: pointer; min-height: var(--hit);
                            display: flex; align-items: center; }
  .proven-rows { list-style: none; margin: 0; padding: 0; }
  .proven-row { display: grid; grid-template-columns: 2.4em minmax(0, 1fr) auto; gap: 8px;
                align-items: center; min-height: var(--hit); padding: 4px 8px;
                border-radius: var(--r-ctl); color: var(--ink); text-decoration: none; }
  .proven-row:hover { background: var(--sunk); }
  .proven-row.on { background: var(--now); }
  .proven-row .n { font-family: var(--f-mono); font-size: 18px; font-weight: 600; }
  .proven-row .what { display: flex; flex-direction: column; min-width: 0; }
  .proven-row .done { font-size: 13px; color: var(--ink); white-space: nowrap; }
  .proven-open { font-weight: 600; }
  /* K1: the start position at the head of the play column, drawn large before any game
     starts; its width is held so a tall board's foot stays on the first screen. */
  .opening { margin: 0; width: 100%; }
  .opening figcaption { margin-top: 6px; }
  .opening .frame-legend { margin-top: 4px; }
  /* K3: a tapped piece's move sentences, one line under the board; and the first game.
     Q809: held to that one line (the line `fitBoard` keeps), the rest a tap away, so a
     piece whose sentences run to three lines pushes nothing off the first screen until
     the reader asks for them; the board is the same size either way. */
  .piece-says { margin: 6px 0 0; display: -webkit-box; -webkit-box-orient: vertical;
                -webkit-line-clamp: 1; overflow: hidden; }
  .piece-says.more { cursor: pointer; }
  .piece-says.open { display: block; -webkit-line-clamp: none; }
  .piece-says[hidden] { display: none; }
  .firstGame { margin: 0 0 10px; }
  .seatCard .gentlest { color: var(--act); font-weight: 600; }
  /* K2: each piece's own small board, under its row. */
  .pieceAlone + .pieceAlone { border-top: 1px solid var(--line); padding-top: 8px; }
  .pieceMap { margin: 0 0 10px; }
  .pieceMap > summary { min-height: var(--hit); display: flex; align-items: center; }
  .pieceBoard { width: min(100%, 220px); margin: 4px 0 6px; }
  /* A section's heading folds it, with the disclosure's own arrow (below). */
  .sect > summary { display: flex; align-items: baseline; gap: 2px; min-height: var(--hit); }
  .sect > summary h3 { margin: 14px 0 6px; }
  .sect > .panel + .panel { margin-top: 10px; }
  svg#board { width: 100%; height: auto; display: block; }
  /* The board's wood (Q588, plan-visual.md §9's third answer). The frame is walnut, its
     grain started where `wood.js` hashed this board's identity to (`--wood-x`, `--wood-y`),
     and about 1.5% of the board's width thick: 8 px round a play board, 2 px round a card's
     thumbnail. Each square's grain is a `.grain` cell over the square's colour and under
     everything else, at the theme's opacity, so the switch re-grains without a redraw. */
  .board-frame { overflow: hidden; border-radius: 5px; position: relative;
                 padding: clamp(2px, 1.5%, 8px);
                 background: var(--walnut) var(--wood-x, 0) var(--wood-y, 0) / 240px, var(--frame); }
  .board-frame svg { display: block; }
  .grain { pointer-events: none; }
  .grain.lt { opacity: var(--grain-lt-a); }
  .grain.dk { opacity: var(--grain-dk-a); }
  /* The play frame (plan-visual.md §3.3; Q577 placed it, V4 gave it its look): status,
     board, controls, legend, in that order and in that place in both chromes, each slot
     named by its `data-frame`. The strip is in ink, never muted (S1): a dot in the colour
     of the side to move, its name, and *in check* as a filled pill in the alarm red; it
     turns into the verdict when the game is over, and the result is put on a banner over
     the board as well. The board panel's width is set by `fitBoard` so the controls and
     the legend under it stay in the first screen (S2).
     The strip's height is fixed, never its content's (Q725: on a phone the board jumped
     whenever *thinking* or *in check* took another line). It holds `--strip-lines` lines
     of 22 px: two, three where `fitBoard` finds the strip narrow (`narrow`), one more for a
     game between two people, whose sentences are longer (`between`). Every state keeps the
     same padding and border, the ended one only colours its border in; and a sentence
     longer than its room ends in an ellipsis rather than growing the box. */
  .frame-status { --strip-lines: 2; height: calc(var(--strip-lines) * 22px + 14px);
                  box-sizing: border-box; overflow: hidden; flex-wrap: nowrap;
                  padding: 6px 0; border: 1px solid transparent;
                  gap: 8px; margin-bottom: 6px; color: var(--ink);
                  font-size: 16px; line-height: 22px; font-weight: 600; }
  .frame-status.narrow, .frame-status.between { --strip-lines: 3; }
  .frame-status.between.narrow { --strip-lines: 4; }
  .frame-status #status { display: -webkit-box; -webkit-box-orient: vertical;
                          -webkit-line-clamp: var(--strip-lines); overflow: hidden; min-width: 0; }
  .frame-status.ended { padding: 6px 12px; border-radius: var(--r-panel); border-color: var(--ink); }
  .frame-status .frame-note { font-weight: 400; margin-left: 10px; }
  /* Q733: the slot under the board of a game between two people, what to do next in the
     frame's largest words. Its height is fixed like the strip's, so the board never moves
     as it changes; a move picked and not sent dashes its border in ink, as the move is
     dashed on the board, and *Send your move* is the frame's one big button. */
  .corr-send { height: 140px; box-sizing: border-box; overflow: hidden; margin-top: 8px;
               padding: 8px 12px; display: flex; flex-direction: column; justify-content: center;
               gap: 8px; border: 2px solid var(--line); border-radius: var(--r-panel);
               background: var(--panel); color: var(--ink); font-size: 16px; line-height: 22px; }
  .corr-send[hidden] { display: none; }
  .corr-send.held { border-style: dashed; border-color: var(--ink); }
  .corr-send .send-says { display: -webkit-box; -webkit-box-orient: vertical;
                          -webkit-line-clamp: 2; overflow: hidden; }
  .corr-send .send-row { display: flex; gap: 10px; align-items: center; flex-wrap: nowrap; }
  /* Q761: a button too narrow for its words wraps them onto a second line rather than
     clipping them; each keeps at least its longest word (a flex item's own minimum), and
     `fitBoard` gives the slot the width the two need side by side (`SEND_WIDE`). */
  .corr-send button.send-big { flex: 0 1 auto; min-height: var(--ctl-h-lg); padding: 8px 24px;
                               font-size: 19px; line-height: 22px; font-weight: 700; }
  /* Q740: *Change my move* beside it, a plain outlined button as tall as *Send your move*,
     edged in ink so it reads as a button at a glance and not as a word on the panel. */
  .corr-send button.send-change { flex: 0 1 auto; min-height: var(--ctl-h-lg); padding: 8px 16px;
                                  font-size: 17px; line-height: 20px; font-weight: 600;
                                  border: 2px solid var(--ink); color: var(--ink); }
  /* Q761: the slot beside the board, where the column has room for it next to a board the
     screen's height holds short (`fitBoard`'s `beside`). It stands at the board's foot, as
     tall as its words and buttons, which take as many lines as they need: the board's own
     height is what the row is, so nothing here moves the board or the bar under it. */
  .board-row.beside { display: flex; align-items: flex-end; gap: 16px; }
  .board-row.beside > .board-panel { flex: 0 0 auto; }
  .board-row.beside > .corr-send { flex: 1 1 0; min-width: 0; height: auto; min-height: 140px;
                                   margin-top: 0; }
  .board-row.beside .send-row { flex-wrap: wrap; }
  .board-row.beside .send-says, .board-row.beside .send-says:only-child { display: block; }
  .corr-send button.send-change:hover:not(:disabled) { border-color: var(--ink);
                                                       background: var(--sunk); }
  .corr-send .send-says:only-child { -webkit-line-clamp: 4; }
  .held-note { margin-top: 4px; font-size: 15px; }
  .seat-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%;
              margin-right: 8px; vertical-align: -1px; border: 1.5px solid var(--ink); }
  .seat-dot.s0 { background: var(--p0); }
  .seat-dot.s1 { background: var(--p1); }
  .pill { display: inline-block; font-size: 12px; font-weight: 600; line-height: 1.5;
          padding: 0 8px; border-radius: 20px; border: 1px solid currentColor; }
  .pill.alarm { color: var(--alarm-ink); background: var(--alarm); border-color: var(--alarm); }
  .frame-legend { color: var(--muted); margin: 8px 0 0; display: flex; flex-wrap: wrap;
                  gap: 4px 12px; align-items: center; }
  .frame-legend[hidden] { display: none; }
  .frame-legend #legend { display: inline-flex; flex-wrap: wrap; gap: 4px 12px; }
  .frame-legend .key { vertical-align: -3px; margin-right: 4px; border-radius: 2px; }
  .frame-legend #zonetoggle, .frame-legend #linktoggle { min-height: var(--ctl-h-sm);
                              padding: 3px 10px; font-size: 13px; line-height: 18px; }
  .frame-legend #zonetoggle + #linktoggle { margin-left: 8px; }
  /* The square names stand outside the board, in the panel's margin, in the numbers'
     face (§3.3): `drawCoords` places each at its edge as a share of the drawing, and it is
     set off outwards by `--co-out`, the frame's wood and a little more. A board that keeps
     its names on its squares (a wrapping one, among its ghosts) has no margin for them. */
  .board-panel { margin: 0; --co-out: 17px; }
  .board-panel.coords-out { padding-left: 26px; padding-bottom: 28px; }
  .board-panel .board-frame { overflow: visible; }
  /* Q809: the designer's bench board names its squares the same way, in a margin of its
     own (`edgeNamesHtml`). */
  .still-named { padding-left: 26px; padding-bottom: 28px; --co-out: 17px; }
  .still-named .board-frame { overflow: visible; }
  .board-box { position: relative; }
  .coords { position: absolute; inset: 0; pointer-events: none; }
  .coords .co { position: absolute; font: 600 11px/1 var(--f-mono); color: var(--muted);
                white-space: nowrap; user-select: none; font-variant-numeric: tabular-nums;
                transform: translate(-50%, -50%)
                  translate(calc(var(--dx) * var(--co-out)), calc(var(--dy) * var(--co-out))); }
  .coords.on-wood { --co-out: 8px; }
  .coords.on-wood .co { background: var(--panel); border-radius: 3px; padding: 2px 3px; }
  /* The ended state: a banner over the board's middle, on a sheet of the panel's paper,
     the board visible round it. The result in the heading face, in the reader's own win
     or loss colour where the reader had a side; the engine's reason under it. */
  .frame-ended { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
                 z-index: 2; width: min(80%, 360px); padding: 16px 18px 14px; text-align: center;
                 background: var(--paper) 0 0 / 180px, var(--panel); color: var(--ink);
                 border: 2px solid var(--ink); border-radius: var(--r-panel);
                 box-shadow: var(--shadow-pop); }
  .frame-ended[hidden] { display: none; }
  .frame-ended .ended-head { font: 600 24px/30px var(--f-head); letter-spacing: -.01em; }
  .frame-ended .ended-how { color: var(--muted); margin-top: 2px; }
  .frame-ended .row { justify-content: center; margin-top: 12px; gap: 6px; }
  .frame-ended.won { border-color: var(--win); }
  .frame-ended.won .ended-head { color: var(--win); }
  .frame-ended.lost { border-color: var(--loss); }
  .frame-ended.lost .ended-head { color: var(--loss); }
  /* What a promoting piece becomes (Q765): the board dimmed under `--scrim`, and over its
     middle a card of the panel's paper saying what to do, each piece drawn large on a
     light square of the board's wood with its name under it, every one a big target. The
     card may be wider than a small board; it is never wider than the screen. Every button
     in it is a large control (52 px, 56 under a finger). */
  .promo { position: absolute; inset: 0; z-index: 3; background: var(--scrim);
           border-radius: inherit; scroll-margin: 12px; }
  .promo[hidden] { display: none; }
  .promo-card { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%);
                width: max-content; max-width: min(440px, calc(100vw - 24px));
                padding: 16px 18px 12px; text-align: center; color: var(--ink);
                background: var(--paper) 0 0 / 180px, var(--panel);
                border: 2px solid var(--ink); border-radius: var(--r-panel);
                box-shadow: var(--shadow-pop); scroll-margin: 12px; }
  .promo-head { font: 600 22px/28px var(--f-head); margin: 0; letter-spacing: -.01em; }
  .promo-says { margin: 4px 0 12px; font-size: 16px; line-height: 22px; }
  .promo-pieces { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }
  .promo-pieces .promo-pick { display: inline-flex; flex-direction: column; align-items: center;
                              gap: 4px; width: 92px; min-height: 0; padding: 6px 4px 8px;
                              font-size: 16px; line-height: 20px; font-weight: 600;
                              white-space: normal; overflow-wrap: anywhere; }
  .promo-pick .promo-piece { width: 72px; height: 72px; border-radius: 6px;
                             background: var(--sq-lt); border: 1px solid var(--edge); }
  /* Where a strike's survivor goes (Q769): each square a light square of the board's wood
     the size of a drawn piece, its name in the frame's brown, which reads on the wood in
     both themes. */
  .promo-sq { display: flex; align-items: center; justify-content: center; width: 72px;
              height: 72px; border-radius: 6px; background: var(--sq-lt);
              border: 1px solid var(--edge); color: var(--frame);
              font: 700 26px/1 var(--f-head); letter-spacing: .01em; }
  /* Which way a piece goes (schema 19, Q832): the piece in the tile, as the promotion draws
     it, and the way below it in the body's own words and size, broken only between words,
     wide enough that a portal's whole name (number, colour, square) takes three lines. Two
     ways side by side fit a 375 px screen's card. */
  .promo-pieces .promo-pick.promo-route { width: 156px; font-weight: 400;
                                          overflow-wrap: break-word; }
  .promo-route .promo-way { font-size: 16px; line-height: 21px; }
  .promo-route .promo-text { font-weight: 600; }
  .promo-row { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
  .promo-card .promo-row { margin-top: 12px; }
  .promo-row button { min-height: var(--ctl-h-lg); font-size: 16px; }
  /* Looking at the board while choosing: no dimming and nothing over the squares, and one
     bar just under the board (over the square names and the line below them), which takes
     the way back to the choice and Cancel. The board still takes no tap. */
  .promo.looking { background: transparent; pointer-events: none; }
  .promo-bar { position: absolute; left: 50%; top: calc(100% + 6px); transform: translateX(-50%);
               pointer-events: auto; width: max-content; max-width: min(440px, calc(100vw - 24px));
               display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
               gap: 6px 12px; padding: 8px 12px; text-align: center; color: var(--ink);
               background: var(--paper) 0 0 / 180px, var(--panel); border: 2px solid var(--ink);
               border-radius: var(--r-panel); box-shadow: var(--shadow-pop); scroll-margin: 12px; }
  .promo-wait { font-size: 16px; line-height: 22px; font-weight: 600; }
  /* The way out of a portal (schema 19, `docs/archive/plan-portals.md` D6, N6): the bar under the
     board, the board left undimmed so the exit each button names is lit on it as the
     button is hovered or focused. One button a way, its arrow and its square in words. */
  .promo-exits { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
  .promo-exits .promo-exit { min-height: var(--ctl-h-lg); font-size: 16px; }
  .promo-exit .promo-arrow { font-size: 20px; margin-right: 6px; }
  .promo-exit-bar > * { flex: 1 1 100%; }
  .promo-exit-bar .promo-row { align-items: center; }
  .promo-esc { font-size: 13px; }
  /* A portal end (Q838b, `docs/archive/plan-portal-look.md` §2.3, the vortex; Q838f). On a live
     board its arms turn inwards, once every 14 s, on the arms' own group so the trip's
     pulse on the glyph composes with it; never under reduced motion. An end nobody may go
     in by has no arms, so nothing on it turns. Drawn small (D9): the token and the sunk
     hole stay at every size; under 28 px a cell the arms' edges are held at one screen
     pixel (scaled, they would vanish, and a black arm on violet with them), under 20 px
     the arms give way to two broad wedges, edged the same (a thumbnail's `pg-small` too),
     and the label and the line under the board say the rest. Q838h: at the mid size and on
     a phone (`data-phone`) the arms are drawn by mass (`pg-mass`: wider, a white arm edged
     in the group's colour), so a white arm's dark edge never reads as a dark arm. Q843b: the
     beak where an end leads stands outside the well at the full size (`pg-beak`), is a wedge
     cut into the rim at the mid size (`pg-beak-cut`) and is not drawn smaller. */
  .pg-idle { animation: pg-turn 14s linear infinite; }
  @keyframes pg-turn { to { transform: rotate(-360deg); } }
  @media (prefers-reduced-motion: reduce) { .pg-idle { animation: none; } }
  .pg-wedge, .pg-mass, .pg-beak-cut { display: none; }
  svg[data-portals="mid"] .pg-arm, svg[data-phone] .pg-arm { display: none; }
  svg[data-portals="mid"] .pg-mass, svg[data-phone] .pg-mass {
    display: inline; vector-effect: non-scaling-stroke; stroke-width: 1px; }
  svg[data-portals="mid"] .pg-beak { display: none; }
  svg[data-portals="mid"] .pg-beak-cut {
    display: inline; vector-effect: non-scaling-stroke; stroke-width: 1px; }
  svg[data-portals="small"] .pg-arm, .pg-small .pg-arm,
  svg[data-portals="small"] .pg-mass, .pg-small .pg-mass,
  svg[data-portals="small"] .pg-beak, .pg-small .pg-beak { display: none; }
  svg[data-portals="small"] .pg-wedge, .pg-small .pg-wedge {
    display: inline; vector-effect: non-scaling-stroke; stroke-width: 1px; }
  /* A fold portal's pool (Q843d, `docs/archive/plan-portals-2.md` §5.3): on a live board its bands
     turn once every 12 s (inwards, or outwards where it leads nowhere) and its rim churns
     once every 8 s, a group's ends in step; never under reduced motion. Its beak takes the
     vortex's classes, so the same sizes show it. Small, the bands' edges are held at two screen
     pixels, so a white band still shows on the pool's light middle. */
  .pf-idle { animation: pf-turn 12s linear infinite; }
  .pf-idle[data-turn="out"] { animation-name: pf-turn-out; }
  .pf-wobble { animation: pf-wobble 8s linear infinite; }
  @keyframes pf-turn { to { transform: rotate(-360deg); } }
  @keyframes pf-turn-out { to { transform: rotate(360deg); } }
  @keyframes pf-wobble { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .pf-idle, .pf-wobble { animation: none; } }
  svg[data-portals="mid"] .pf-band, svg[data-phone] .pf-band, svg[data-portals="small"] .pf-band {
    vector-effect: non-scaling-stroke; stroke-width: 2px; }
  .frame-controls { margin-top: 8px; gap: 6px; }
  .frame-controls .shift { gap: 6px; }
  .frame-controls .shift:empty { display: none; }
  .frame-controls #resign { margin-left: auto; }
  .frame-controls .ico { display: none; }
  /* On a phone the bar is one row of icons, a word under each, every one a 44 px target
     or more (§3.3, §3.5). A wrapping board's arrows take a row of their own under it, and
     the sentence naming the repeat on screen goes (the reset button's title still says
     it). The banner shrinks to the phone's board. */
  @media (max-width: 480px) {
    .frame-controls { flex-wrap: wrap; gap: 4px; }
    .frame-controls > button { flex: 1 1 0; min-width: 0; min-height: var(--ctl-h);
                               display: inline-flex; flex-direction: column; align-items: center;
                               justify-content: center; gap: 2px; padding: 5px 2px;
                               font-size: 12px; line-height: 14px; }
    .frame-controls .ico { display: block; }
    .frame-controls #resign { margin-left: 0; }
    .frame-controls .shift { order: 1; flex: 1 0 100%; }
    .frame-controls #frame { display: none; }
    .frame-controls .shift button { padding-left: 9px; padding-right: 9px; }
    /* A hidden button stays hidden (Q733: *rematch* showed on a phone in a game still being
       played, its `display` winning over `hidden`). */
    .frame-controls > button[hidden] { display: none; }
    /* Q733: on a phone the slot that sends a move is its button row alone while a move is
       held, the strip right above the board saying it is not sent yet, and the legend's
       *on the board:* runs on with its first entry; both so the frame of a game between
       two people still ends inside the first screen. */
    .corr-send { height: calc(var(--ctl-h-lg) + 12px); padding: 4px 10px; margin-top: 6px; }
    .corr-send .more { display: none; }
    /* Q740: both buttons on one row of a phone. Q761: *Send your move* keeps its one line
       where it can and *Change my move* takes the rest, two lines inside its height; on a
       screen narrower still both wrap, and neither is ever narrower than its longest word. */
    .corr-send .send-row { gap: 8px; }
    .corr-send button.send-big { flex: 0 1 auto; padding: 4px 14px; font-size: 18px;
                                 line-height: 21px; }
    .corr-send button.send-change { flex: 1 1 0; padding: 4px 8px;
                                    font-size: 16px; line-height: 19px; }
    .corr-send.held .send-says { display: none; }
    .corr-send .send-says, .corr-send .send-says:only-child { -webkit-line-clamp: 2;
                                                             font-size: 15px; line-height: 20px; }
    .frame-legend #legend { display: block; }
    .board-panel { --co-out: 14px; }
    .board-panel.coords-out { padding-left: 20px; padding-bottom: 22px; }
    .still-named { padding-left: 20px; padding-bottom: 22px; --co-out: 14px; }
    .frame-ended { padding: 10px 12px; }
    .frame-ended .ended-head { font-size: 19px; line-height: 24px; }
    .frame-ended .row { margin-top: 8px; }
    .promo-card { padding: 12px 10px 10px; }
    .promo-head { font-size: 20px; line-height: 25px; }
    .promo-pieces { gap: 8px; }
    .promo-pieces .promo-pick { width: 72px; font-size: 15px; }
    .promo-pick .promo-piece, .promo-pick .promo-sq { width: 56px; height: 56px; }
    .promo-pieces .promo-pick.promo-route { width: 148px; font-size: 16px; }
  }
  /* The board alone (Q638, `board.js` `setBoardFull`): the play frame as a view of its own
     over the page, on the page's paper, holding only what a move needs: the strip (whose
     move, the result), the board with its banner and its cards, the slot a round or a game
     between two people sends its move from, the hand of a game with drops, and the way out.
     Upright, the strip and the way out share the top row and the rest stand under the
     board; on its side (`.wide`, which `fitFull` sets from the window) the board takes the
     height and the rest stand in a column beside it. Once the game is over the review row
     joins them, since the banner's *review* steps back through the game. `fitFull` sizes the board to the room
     left, so the rows carry margins rather than gaps: an empty slot or hand takes nothing.
     The control is a phone's and a tablet's (or a narrow window's); at a desk it is not
     offered, except to leave a view a window was narrow for when it opened. */
  @media (min-width: 861px) and (hover: hover) and (pointer: fine) {
    html:not(.board-full) .frame-controls #fullBoard { display: none; }
  }
  html.board-full, html.board-full body { overflow: hidden; overscroll-behavior: none; }
  html.board-full .play-frame {
    position: fixed; inset: 0; z-index: 60; margin: 0; box-sizing: border-box; overflow: hidden;
    display: grid; align-content: center; gap: 0 12px;
    grid-template-columns: minmax(0, 1fr) auto;
    grid-template-areas: "status bar" "board board" "send send" "hand hand" "review review";
    padding: max(8px, env(safe-area-inset-top)) max(10px, env(safe-area-inset-right))
             max(8px, env(safe-area-inset-bottom)) max(10px, env(safe-area-inset-left));
    background: var(--paper) 0 0 / 180px, var(--bg);
  }
  html.board-full .play-frame > .row:not([data-frame]), html.board-full .play-frame > #ground,
  html.board-full .play-frame > .piece-says, html.board-full .play-frame > .frame-legend {
    display: none; }
  html.board-full .play-frame > .frame-status { grid-area: status; margin: 0; max-width: none; }
  html.board-full .play-frame > .board-row { display: contents; }
  html.board-full .play-frame .board-panel { grid-area: board; justify-self: center; margin-top: 8px; }
  html.board-full .play-frame .corr-send { grid-area: send; max-width: none; }
  html.board-full .play-frame > .hand { grid-area: hand; margin-top: 8px; }
  html.board-full .play-frame > .hand:empty { display: none; }
  html.board-full .play-frame > .frame-status.ended ~ #review { display: flex; grid-area: review;
                                                                margin-top: 8px; }
  html.board-full .play-frame > .frame-controls { grid-area: bar; margin: 0; align-self: start;
                                                  flex-wrap: nowrap; }
  html.board-full .frame-controls > :not(#fullBoard) { display: none; }
  html.board-full .frame-controls > #fullBoard { flex: 0 0 auto; min-width: var(--ctl-h); }
  html.board-full .play-frame.wide {
    align-content: stretch; grid-template-columns: minmax(0, 1fr) minmax(200px, 36%);
    grid-template-rows: auto auto auto auto minmax(0, 1fr);
    grid-template-areas: "board status" "board send" "board hand" "board review" "board bar";
  }
  html.board-full .play-frame.wide .board-panel { margin-top: 0; align-self: center; }
  html.board-full .play-frame.wide > .frame-controls { align-self: end; justify-self: end; }
  html.board-full .toast.error { z-index: 70; }
  button.danger { border-color: var(--alarm); color: var(--alarm); }
  .moves div.result { margin-top: 4px; }
  .moves .sign { font-weight: 700; }
  pre { white-space: pre-wrap; font: 13px/1.5 var(--f-mono);
        margin: 0; color: var(--ink); }
  pre.log { max-height: 60vh; overflow: auto; font-size: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  td, th { padding: 3px 0; text-align: left; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted);
           font-family: var(--f-mono); font-size: .95em; }
  .bar { height: 5px; background: var(--line); border-radius: 3px; overflow: hidden; min-width: 60px; }
  .bar > div { height: 100%; background: var(--meter); }
  .moves { max-height: 42vh; overflow-y: auto; font: 14px/20px var(--f-mono); }
  .moves div { padding: 2px 6px; border-radius: 5px; }
  /* Striped by ply rather than by row: a ply with wild pieces is two rows, the move and
     the phase the rules ran after it, and the two read as one band. */
  .moves div.stripe { background: var(--stripe); }
  .moves div.phase { padding-left: 22px; color: var(--muted); font-size: 12px; }
  .moves div.took { color: var(--ink); }
  /* The ply the board is showing, while a game is being stepped through. It wins over
     the striping above it, which is why it is stated after. */
  .moves div.on { background: var(--now); color: var(--ink); font-weight: 600; }
  .hand { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
  /* Generation 42 (N4c): the duck waiting off the board, in the hand's strip. */
  .hand .duck-tray { width: 32px; height: 32px; flex: none; }
  .hand .duck-tray + span { flex: 1 1 0; min-width: 0; }
  .verdict { margin-top: 12px; padding: 10px 12px; border-radius: var(--r-ctl);
             border: 1px solid var(--note); font-weight: 600; }
  /* Ruled out by a gate: the verdict's box in the colour of *bad*. */
  .verdict.out { border-color: var(--bad); }
  .error { border: 1px solid var(--bad); color: var(--bad); border-radius: var(--r-ctl);
           padding: 9px 12px; margin: 12px 0; }
  /* An error shown beside the control that raised it. `.row` and `.hand` wrap, so a
     100% basis gives it a line of its own under the whole control line -- never over
     the button, and never between a button and the sentence explaining it. */
  .error.beside { flex: 1 0 100%; margin: 8px 0 0; }
  /* An error with no control behind it, fixed to the viewport so that scrolling cannot
     carry it off. At the bottom, because the top is the header and whatever was last
     read. It needs a background of its own: it floats over the page. */
  .error.toast { position: fixed; z-index: 30; left: 12px; right: 12px; bottom: 12px;
                 max-width: 440px; margin: 0 auto; background: var(--panel);
                 box-shadow: var(--shadow-toast);
                 display: flex; gap: 10px; align-items: flex-start; }
  .error.toast > button { margin-left: auto; flex: none; padding: 0 4px; border: none;
                          min-height: var(--hit); min-width: var(--hit);
                          background: none; color: var(--bad); font-size: 17px;
                          line-height: 1.2; cursor: pointer; }
  .busy { display: flex; align-items: center; gap: 8px; color: var(--muted);
          padding: 9px 12px; margin: 12px 0; }
  .crumb { background: none; border: none; padding: 0; min-height: var(--hit); color: var(--muted);
           cursor: pointer; font-weight: 400; }
  .crumb:hover { color: var(--ink); }
  .spin { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
          background: var(--muted); animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { opacity: .25; } 50% { opacity: 1; } }

  /* --- disclosure, used everywhere something has a plain answer and a long one --- */
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: '\\25B8'; display: inline-block; width: 1em;
    color: var(--muted); transition: transform .18s ease; }
  details[open] > summary::before { transform: rotate(90deg); }
  .why { margin: 8px 0 4px 1em; padding-left: 12px; border-left: 2px solid var(--line);
         font-size: 13px; color: var(--muted); }
  .why p { margin: 0 0 7px; }
  .why b { color: var(--ink); font-weight: 600; }

  /* --- the map of kinds --- */
  .axes { display: grid; gap: 8px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
          margin-top: 10px; }
  /* A definition grid, apart by rules: no box inside the panel (§3.4). */
  .axis { border-top: 1px solid var(--line); padding: 8px 2px 4px; }
  .axis b { display: block; font-size: 12px; text-transform: uppercase;
            letter-spacing: .07em; color: var(--muted); font-weight: 600; }
  .axis .asks { font-size: 13px; }
  .axis .bins { margin-top: 5px; display: flex; gap: 4px; flex-wrap: wrap; }
  .cells { margin-top: 12px; overflow-x: auto; }
  .cells table { font-size: 12px; min-width: 620px; }
  .cells th { color: var(--muted); font-weight: 600; text-transform: uppercase;
              font-size: 11px; letter-spacing: .06em; border-bottom: 1px solid var(--line);
              padding-bottom: 4px; }
  .cells td { padding: 4px 10px 4px 0; border-bottom: 1px solid var(--line); }
  .cells td.here { color: var(--ink); }
  .cells tr:hover td { background: var(--sunk); }

  /* --- pieces --- */
  .pieces { display: grid; gap: 10px; }
  .piece { display: grid; grid-template-columns: 38px 1fr; gap: 10px; align-items: start;
           padding: 8px; border-radius: 9px; }
  .piece.on { background: var(--sunk); outline: 1px solid var(--act); }
  .piece .art { display: flex; gap: 2px; }
  .piece .art img { width: 34px; height: 34px; }
  .piece .art .letter { width: 34px; height: 34px; display: flex; align-items: center;
                        justify-content: center; border-radius: 50%; font-weight: 600;
                        background: var(--p0); color: var(--p0-ink); border: 1px solid var(--ink); }
  .piece h4 { margin: 0; font-size: 14px; }
  .piece ul { margin: 4px 0 0; padding-left: 16px; font-size: 12.5px; color: var(--muted); }
  .royal { font-size: 10px; border: 1px solid var(--note); color: var(--ink);
           border-radius: 20px; padding: 1px 6px; margin-left: 6px; vertical-align: 1px; }
  /* Schema 28: a piece's speed on its card, its meaning in the hover line. */
  .speed { font-size: 10px; border: 1px solid var(--line); color: var(--ink);
           border-radius: 20px; padding: 1px 6px; margin-left: 6px; vertical-align: 1px; }

  /* --- choosing a side --- */
  /* One row each, apart by a rule rather than a box inside the play panel (§3.4). */
  .sides { display: grid; gap: 0; grid-template-columns: minmax(0, 1fr); }
  .side { padding: 10px 0; min-width: 0; }
  .side + .side { border-top: 1px solid var(--line); }
  .side select { width: 100%; margin-top: 5px; }
  /* Who takes the seat, as cards (F7's look, §3.5): the word, its strength under it, the
     chosen one bordered in the action colour. A busy player's line is muted, never red. */
  .seatCards { display: grid; gap: 6px; margin-top: 6px;
               grid-template-columns: repeat(auto-fill, minmax(104px, 1fr)); }
  .seatCard { display: flex; flex-direction: column; align-items: flex-start; gap: 1px;
              padding: 6px 9px; text-align: left; line-height: 18px; }
  .seatCard b { font-weight: 600; }
  .seatCard span { font-size: 12px; line-height: 15px; }
  .seatCard[aria-pressed="true"] { border-color: var(--act); box-shadow: inset 0 0 0 1px var(--act); }
  .seatCard[aria-pressed="true"]:hover:not(:disabled) { border-color: var(--act); }
  /* On a phone each seat's cards are one row that scrolls inside itself, so the start
     button stays near the top; the page itself never scrolls sideways. */
  .seatCards { position: relative; }
  @media (max-width: 480px) {
    .seatCards { grid-auto-flow: column; grid-template-columns: none;
                 grid-auto-columns: minmax(92px, max-content); overflow-x: auto;
                 overscroll-behavior-x: contain; scroll-snap-type: x proximity;
                 padding-bottom: 4px; }
    .seatCard { scroll-snap-align: start; }
  }
  .side .army { margin-top: 6px; }
  /* The sentence behind a rule of thumb. It wraps, because a plan is a sentence and
     truncating it would leave the name carrying the whole idea again. */
  .side .plan:not(:empty) { margin-top: 6px; font-style: italic; line-height: 1.35; }

  /* --- how strong they are here --- */
  table.ratings { width: 100%; border-collapse: collapse; margin-top: 8px; }
  table.ratings th { text-align: left; font-weight: 600; padding: 3px 4px;
    border-bottom: 1px solid var(--line); }
  table.ratings td { padding: 3px 4px; vertical-align: top; }
  /* An opponent with no rating table row is named by its parameters, and
     `minimax:net=0eaad64c4586,nodes=1600` is one word to a browser. Unbroken, it set the
     table's width and the phone scrolled sideways. */
  table.ratings td, table.ratings th { overflow-wrap: anywhere; }
  table.ratings tbody tr:nth-child(odd) { background: var(--stripe); }
  /* The reader's own row on a game's leaderboard (L2): in weight, and said in words. */
  table.ratings tr.you td { font-weight: 600; }

  /* --- rules --- */
  ul.rules { margin: 0; padding-left: 18px; }
  ul.rules li { margin-bottom: 5px; }

  /* --- the score panel --- */
  .comp { border-top: 1px solid var(--line); padding: 7px 0; }
  .comp:first-of-type { border-top: none; }
  /* Two lines, not four columns. A 380px panel cannot give a name, a raw value, a bar
     and a score a column each without wrapping every one of them. */
  /* Three columns, because the disclosure triangle is a ::before on the summary and
     therefore a grid item like any other -- given two, it takes the first and pushes
     the name into the value's column. */
  .comp summary { display: grid; grid-template-columns: 1em 1fr auto; gap: 2px 8px;
                  align-items: baseline; font-size: 13px; }
  .comp .scored { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-weight: 600; }
  /* The value may wrap: on the site a unit can be a phrase (*166 rating points each time
     thinking time doubles*), which held on one line pushed the bar past a 375px screen. */
  .comp .meta { grid-column: 2 / -1; display: grid;
                grid-template-columns: minmax(0, max-content) minmax(40px, 1fr);
                gap: 8px; align-items: center; }
  .comp .raw { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .curve { margin: 8px 0 2px; }
  .curve svg { width: 100%; height: 74px; display: block; }
  .fam { display: grid; grid-template-columns: 92px 1fr 38px; gap: 8px; align-items: center;
         font-size: 13px; padding: 3px 0; }
  .gate { font-size: 12.5px; color: var(--muted); padding: 3px 0 3px 14px; text-indent: -14px; }

  /* --- the (i) beside a term, and the one popover it opens --- */
  button.info {
    width: 15px; height: 15px; min-height: 0; padding: 0; margin-left: 5px; border-radius: 50%;
    font-size: 10px; line-height: 1; font-weight: 700; font-style: italic;
    color: var(--muted); border-color: var(--line); background: transparent;
    vertical-align: 1px; display: inline-flex; align-items: center; justify-content: center;
  }
  button.info:hover { color: var(--ink); border-color: var(--muted); }
  .bubble {
    position: absolute; z-index: 40; background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
    box-shadow: var(--shadow-pop); font-size: 13px; line-height: 1.45;
    max-height: 70vh; overflow-y: auto;
  }
  .bubble b { display: block; margin-bottom: 4px; }
  .bubble p { margin: 0; color: var(--muted); }
  .bubble p + p, .bubble details p, .bubble details { margin-top: 6px; }
  .bubble summary { cursor: pointer; color: var(--ink); }
  /* The portal's (i), Q838c, in rows since Q838f: what an end stands on, four drawings in a
     row each over its sentence; then its arms, the same; then one pair, its two ends beside
     their sentence. Each drawn on the board's light square. */
  .portal-figure { display: grid; gap: 6px; }
  .portal-head { font-weight: 600; font-size: 12px; margin-top: 4px; }
  .portal-row { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; }
  .portal-example { display: flex; flex-direction: column; align-items: center; gap: 4px;
    text-align: center; font-size: 12px; line-height: 1.3; }
  .portal-row.portal-pairs { grid-template-columns: 1fr; }
  .portal-pairs .portal-example { flex-direction: row; text-align: left; gap: 10px; }
  .portal-example svg { border-radius: 4px; flex: none; }
  .portal-example span { color: var(--muted); }

  /* --- narrowing and ordering the library --- */
  .controls { display: grid; gap: 10px; margin-bottom: 14px; }
  .controls .line { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .controls input[type=search] { flex: 1 1 220px; min-width: 180px; }
  .controls label.pick { display: inline-flex; align-items: center; gap: 6px;
                         font-size: 13px; color: var(--muted); }
  .controls select { max-width: 260px; }
  .chips { display: flex; gap: 5px; flex-wrap: wrap; }
  .chip { font-size: 13px; line-height: 18px; min-height: var(--chip-h); padding: 4px 11px;
          border-radius: 999px; cursor: pointer; font-weight: 400;
          border: 1px solid var(--line); background: var(--panel); color: var(--muted); }
  .chip:hover { border-color: var(--muted); color: var(--ink); }
  .facet { display: grid; grid-template-columns: 116px 1fr; gap: 8px; align-items: start;
           padding: 5px 0; }
  .facet > b { font-size: 12px; font-weight: 600; color: var(--muted);
               text-transform: uppercase; letter-spacing: .06em; padding-top: 4px; }
  .groupHead { display: flex; align-items: baseline; gap: 8px; margin: 20px 0 8px;
               padding-bottom: 5px; border-bottom: 1px solid var(--line); }
  .groupHead h3 { margin: 0; text-transform: none; letter-spacing: 0; font-size: 14px;
                  color: var(--ink); }
  .card .metric { margin-top: 7px; font-size: 12px; color: var(--muted);
                  display: flex; gap: 7px; align-items: center; }
  .card .metric .bar { flex: 1; }
  .empty { padding: 26px; text-align: center; color: var(--muted); }

  /* --- the designer --- */
  .designer { display: grid; gap: 20px; grid-template-columns: minmax(320px, 1fr) minmax(300px, 400px); }
  @media (max-width: 900px) { .designer { grid-template-columns: 1fr; } }
  /* --- the workbench: one large board in the middle of the game designer ---
     Three columns where there is room: the form on the left, the board and its tray in
     the middle, the verdict and the rules in English on the right. The columns are
     placed by `order` rather than by DOM order, because the board is what a phone should
     meet first and the form is what a wide screen should meet first. */
  .designer.bench > #boardSide { order: 1; }
  .designer.bench > #form { order: 2; }
  .designer.bench > #verdictSide { order: 3; }
  @media (min-width: 1200px) {
    .designer.bench { grid-template-columns: minmax(280px, 350px) minmax(380px, 1fr)
                                             minmax(280px, 370px); }
    .designer.bench > #form { order: 1; }
    .designer.bench > #boardSide { order: 2; }
  }
  /* Between the two: the board and the verdict share the width and the form runs full
     width under them, which keeps the board large rather than squeezing three columns
     into a laptop. */
  @media (min-width: 901px) and (max-width: 1199px) {
    .designer.bench { grid-template-columns: minmax(320px, 1fr) minmax(280px, 360px); }
    .designer.bench > #verdictSide { order: 2; }
    .designer.bench > #form { order: 3; grid-column: 1 / -1; }
  }
  .benchPanel { padding-top: 10px; }
  /* A game played here is the play page's board, and that board brings its own panel.
     Inside the workbench's panel that is two frames of padding round one board, which
     left it 271 px wide on a 375 px phone (Q532); the workbench's frame is the one kept. */
  #benchPlay .panel { padding: 0; border: 0; background: none; box-shadow: none; }
  /* A piece just added that no square holds (`freshNoteHtml`). Inside the piece's sheet,
     so the note's rule rather than a box of its own (V8: no panel inside a panel). */
  .freshNote { border-left: 3px solid var(--note); padding: 3px 0 3px 9px; margin: 8px 0; }
  /* PF: what a ready-made piece needs, the rule it would write shown whole, wrapped. */
  .readyNeeds code { white-space: normal; overflow-wrap: anywhere; }
  .readyNeeds > div + div { margin-top: 4px; }
  .benchTabs { gap: 6px; margin-bottom: 10px; }
  .benchTabs button { min-height: var(--ctl-h-sm); padding: 3px 11px; line-height: 18px; }
  /* The tray stands to the board's left, and drops above it when the column is too
     narrow for both -- a board squeezed to make room for a strip of icons is the thing
     this whole layout exists to stop. */
  .benchRow { display: flex; gap: 10px; align-items: flex-start; }
  .benchBoard { flex: 1 1 auto; min-width: 0; max-width: 620px; margin: 0; }
  .benchBoard.stale { opacity: .55; }
  .tray { display: flex; flex-direction: column; gap: 6px; flex: none; }
  .tray .palette { flex-direction: column; gap: 4px; margin: 0; }
  .tray .trayWho { font-size: 11px; }
  @media (max-width: 760px) {
    .benchRow { flex-direction: column; }
    .tray { flex-direction: row; flex-wrap: wrap; width: 100%; }
    .tray .palette { flex-direction: row; }
  }
  /* A hex draft's cells are small at a square board's size (Q545): on a computer its
     board takes the larger share of the three columns and the tray goes above it, so the
     board has the whole column's width. A phone is as it was. */
  .designer.bench.hexBench .benchRow { flex-direction: column; }
  .designer.bench.hexBench .tray { flex-direction: row; flex-wrap: wrap; width: 100%; }
  .designer.bench.hexBench .tray .palette { flex-direction: row; }
  .designer.bench.hexBench .benchBoard { width: 100%; max-width: 720px; }
  /* Portals in the designer (P3, docs/archive/plan-portals.md D2-D9): the status line a screen
     reader reads out, the board taking the keyboard, a transit's legs shown on hover, a
     refused square outlined once, the tray's portal row and the pair's card. */
  .benchSaid { margin-top: 6px; color: var(--ink); }
  .benchSaid:empty, .benchHover:empty { display: none; }
  .benchHover { margin-top: 4px; }
  figure.benchBoard:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  .benchLeg { display: none; }
  .benchLeg.on { display: inline; }
  .benchFlash { animation: benchFlash 0.9s ease-out 1 forwards; }
  @keyframes benchFlash { from { opacity: 1; } to { opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { .benchFlash { animation: none; } }
  .benchLegend { margin-top: 6px; }
  .portalRow { flex-wrap: wrap; align-items: center; }
  .tray .portalRow { margin-top: 2px; padding-top: 6px; border-top: 1px solid var(--line); }
  .portalRow .portalChip { padding: 1px 3px; }
  .portalRow .portalChip svg { display: block; }
  .portalRefused { font-size: 11px; max-width: 14em; }
  /* Beside the board, the tray is a column as wide as its widest row, and the refusal's
     14em made it that wide on every draft with wild pieces, which squeezed a chess board
     at 1400 px to 153 px (Q854). There the sentence wraps to the buttons' width instead. */
  @media (min-width: 761px) {
    .designer.bench:not(.hexBench) .tray .portalRefused { contain: inline-size; min-width: 7em; }
  }
  .tray .wildRow { margin-top: 2px; padding-top: 6px; border-top: 1px solid var(--line);
                   flex-wrap: wrap; align-items: center; }
  .wildTypes { flex-wrap: wrap; align-items: center; }
  .linkish { border: 0; background: none; padding: 0; min-height: 0; color: var(--ink);
             text-decoration: underline; cursor: pointer; font: inherit; }
  .pairCard { border: 1px solid var(--line); border-radius: var(--r-ctl); padding: 8px 10px; margin-top: 8px; }
  .pairCard.on { border-color: var(--act); box-shadow: 0 0 0 1px var(--act); }
  /* A group's ways of moving: a card of its own right under the group's, named by its icon
     and number, so the group's card still fits a screen (Q1014). */
  .pairMoves { border: 1px solid var(--line); border-radius: var(--r-ctl); padding: 8px 10px; margin-top: 4px; }
  .pairMovesHead { display: flex; align-items: center; gap: 6px; }
  .pairMovesHead svg { flex: none; display: block; }
  .pairHead { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .pairHead .grow { flex: 1 1 0; }
  .pairPick { padding: 0; border: 0; background: none; min-height: 0; line-height: 0; }
  .pairPart { margin-top: 7px; }
  .pairSays { color: var(--ink); }
  .pairError { margin-top: 6px; }
  .pairStale { opacity: .55; }
  .cardWarn { border-left: 3px solid var(--note); padding: 2px 0 2px 8px; margin-top: 6px; }
  .wayChoices { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin: 6px 0; }
  .wayChoice { display: flex; flex-direction: column; align-items: center; gap: 2px;
               padding: 6px 4px; font-size: 12px; line-height: 1.25; white-space: normal; }
  .wayChoice.on { border-color: var(--act); background: var(--act-tint); color: var(--ink); font-weight: 600; }
  .wayChoice svg { max-width: 100%; height: auto; }
  .wayFor { margin-top: 6px; }
  .moverRow { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; margin-top: 4px; }
  .moverEnd { min-width: 11em; }
  .moverRadios { display: inline-flex; flex-wrap: wrap; gap: 2px 10px; }
  .moverMenu { display: none; }
  .portalKey { display: flex; gap: 10px; align-items: flex-start; margin-top: 8px;
               padding: 4px 0 4px 9px; border-left: 3px solid var(--note); }
  .portalKeyGrid { display: grid; grid-template-columns: repeat(3, 1.4em); text-align: center;
                   line-height: 1.4em; color: var(--muted); flex: none; }
  .portalKeyGrid .on { color: var(--ink); font-weight: 700; }
  .pickLine { margin-top: 6px; gap: 6px; }
  /* A group's shape, where each end leads and which pieces may go in (schema 21, Q843b,
     `docs/archive/plan-portals-2.md` §5.4): the three named shapes as drawn buttons, *your own map*
     beside them when the rows make none of them, a row per end, and the pieces' icons with
     a tick each when an end lets only some in. */
  .shapeChoices { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  /* How it works (schema 22, Q843d, §5.4): *momentum* or *fold*, each a small button with its
     glyph and its word, on the line above the shapes, so a group's card still fits a screen;
     44 px targets on a phone. */
  .kindLine { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; }
  .kindChoices { display: inline-flex; gap: 6px; }
  .kindChoice { display: inline-flex; align-items: center; gap: 4px; padding: 0 8px; min-height: 22px; }
  .kindChoice.on { border-color: var(--act); background: var(--act-tint); color: var(--ink); font-weight: 600; }
  .kindChoice svg { flex: none; display: block; }
  .shapeOwn { border: 1px solid var(--act); border-radius: var(--r-ctl); justify-content: center; }
  .leadsRow { margin-top: 5px; padding-top: 5px; border-top: 1px solid var(--line); }
  .leadsRow select { margin-left: 4px; min-height: var(--ctl-h-sm); padding: 3px 6px; max-width: 13em; }
  .leadsMore { display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 4px; }
  .filterRow { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; margin-top: 4px; }
  .filterSays { flex-basis: 100%; color: var(--ink); }
  .filterPieces { display: flex; flex-wrap: wrap; gap: 4px; flex-basis: 100%; }
  .filterPiece { display: inline-flex; align-items: center; gap: 2px; padding: 2px 4px; min-width: 44px; }
  .filterPiece img { width: 26px; height: 26px; display: block; }
  .filterPiece.on { border-color: var(--act); background: var(--act-tint); }
  .filterTick { font-weight: 700; min-width: 0.8em; }
  @media (max-width: 760px) {
    .wayChoices { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .shapeChoices { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .kindChoice { min-height: 44px; }
    .wayChoice { min-height: 44px; }
    .moverRadios { display: none; }
    .moverMenu { display: inline; }
    .moverEnd { min-width: 0; }
  }
  @media (min-width: 1200px) {
    .designer.bench.hexBench { grid-template-columns: minmax(260px, 300px) minmax(440px, 1fr)
                                                     minmax(250px, 300px); }
  }
  /* Where to start, above the board (F10): a phone meets it before the board, not two
     screens under it in the form. */
  .startFrom { max-width: 460px; margin: 10px 0 0; }
  /* A rule file written outside the designer (Q830b): the button and the line with the
     link under the chooser, and what became of a file. The validator's lines name paths
     (`pieces[1].moves[0].kind`) and may be long, so they wrap anywhere rather than push a
     phone's page sideways. */
  .ruleFile { max-width: 70ch; margin: 8px 0 0; }
  .ruleFile > .row > button { flex: none; }
  .ruleSaid > .ok, .ruleSaid > .error { margin: 8px 0 0; }
  .ruleSaid code { overflow-wrap: anywhere; }
  .ruleSaid ul.rules { margin-top: 6px; }
  /* Not a game yet, compact (Q580): a heading, the first thing to fix, and the rest one
     tap away. The heading sits in the line, so the bar at the foot of a phone grows by
     three short lines and not by half the screen. */
  .designer .says > .notGame { margin: 0 0 2px; font-size: 15px; }
  .designer .notGameAll > summary { cursor: pointer; width: max-content; margin-top: 4px; }
  .designer .notGameAll > .muted { margin-top: 6px; }
  .designer .notGameAll > .rules { margin: 6px 0 0; }
  .designer .says:has(> .notGameAll[open]) > .notGameFirst { display: none; }
  .checkedAt { white-space: nowrap; }
  /* A group inside the verdict's panel (the ladder, the family): set off by a rule. */
  .ruled { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--line); }
  /* Measuring on this computer, as a step inside *Publish* (`hereHtml`'s `inPanel`): the
     line it answers is in the same panel, so a rule beside it, not a box round it. */
  .hereStep { margin: 8px 0 10px; padding: 2px 0 2px 11px; border-left: 3px solid var(--line); }
  /* A list of marked lines (*Publish*'s two ticks, the shortlist's bar): the mark in a
     column of its own, so a line that wraps on a phone runs on under its words, not
     under the mark (Q777). */
  .marks { list-style: none; padding-left: 0; }
  .marks > li { display: flex; }
  .marks > li > span:first-child { flex: none; width: 1.4em; }
  /* The game as text, in the numbers face. */
  .jsonText { min-height: 220px; margin-top: 8px; font: 12px/1.45 var(--f-mono); }
  /* A typed way of moving, read back in words under its row (F10). */
  .moveSays { margin: 3px 0 2px; }
  .moveSays:empty { display: none; }
  /* Said once, on a phone (`phoneNoteHtml`). */
  .phoneNote { border: 1px solid var(--line); border-radius: var(--r-panel); padding: 8px 10px;
               margin: 8px 0 0; display: flex; gap: 10px; align-items: center; }
  .phoneNote > button { flex: none; white-space: nowrap; }
  /* The board is the single most useful thing this editor draws, and it used to leave
     the screen as soon as you scrolled down to the piece you were editing. The verdict
     column is shorter than the form, so it has travel to spare: the whole column sticks,
     board first. The column sticks rather than the board panel, because a sticky panel
     paints over every panel after it and covers their buttons. `align-self: start` gives
     the column its travel. It has no scrollbar of its own (F11: the wheel over it scrolled
     the column and not the page): a column taller than the window sticks by its foot
     instead, `--stick-top` going negative by what does not fit (`fitSticky`).
     Only where the form has a column of its own, which is the three-column bench. From
     901 to 1199 px the form runs full width under the board and the verdict, so a
     sticking column slid down over it as the page scrolled: at 1024 px, with every panel
     open, 122 of the form's 127 controls came to rest under the board or the verdict,
     *add it* among them (Q490). There the board scrolls away with the page, as stacked. */
  @media (min-width: 1200px) {
    .designer .stick { position: sticky; align-self: start; top: var(--stick-top, 12px); }
  }
  /* Stacked, the verdict column sits under the whole form, and at 375 px that put the
     pass/fail sentence and *save it*, *play it* and *watch it* nine screens down. They
     become the foot bar. Fixed rather than sticky, because a sticky bar inside that
     column would only appear once the form had been scrolled past; `reserveFootbar`
     reads `position` back off the bar and gives the page its height, so the width this
     happens at is written here and nowhere else. Two columns keep it in the panel. */
  /* The verdict's own box, tighter than a standalone `.ok` or `.error` because the panel
     around it is already a box. Written here rather than on the element so that the bar
     can take it back below. */
  .designer .footbar > .says > .ok, .designer .footbar > .says > .error { margin: 10px 0; }
  /* *on its own page*, said once before the two buttons that open it, only in the bar at
     the foot of a phone; in the panel each button says it whole (Q801). */
  .designer .footbar .awayLead { display: none; }
  @media (max-width: 900px) {
    .designer .footbar {
      position: fixed; z-index: 20; left: 0; right: 0; bottom: 0; max-height: 40vh;
      display: flex; flex-direction: column; gap: 6px;
      padding: 4px 14px calc(8px + env(safe-area-inset-bottom));
      background: var(--panel); border-top: 1px solid var(--line);
      box-shadow: var(--shadow-bar);
    }
    /* The verdict is a sentence, or a list of reasons, or every error in the draft, and
       the buttons have to stay on the screen whichever it is -- so the sentence scrolls
       inside the bar and the buttons never move. */
    .designer .footbar > .says { min-height: 0; overflow-y: auto; }
    /* A game that plays is a sentence in the good colour, not a box inside the bar: the
       bar is the box (Q801). The verdict that does not finish keeps its box and scrolls. */
    .designer .footbar > .says > .ok, .designer .footbar > .says > .error { margin: 0; }
    .designer .footbar > .says > .ok { border: none; padding: 2px 0 0; font-size: 14px;
                                       line-height: 1.35; }
    .designer .footbar .awayLead { display: inline; align-self: center; }
    .designer .footbar .awayTail { display: none; }
    /* The first thing to fix, at most two lines in the bar; the whole list is the tap. */
    .designer .footbar .notGameFirst { display: -webkit-box; -webkit-line-clamp: 2;
                                        -webkit-box-orient: vertical; overflow: hidden; }
    .designer .footbar > .row { flex: none; gap: 6px; }
    .designer .footbar button { padding: 6px 10px; }
    /* A control the browser scrolls to -- tabbing through the form, or a field a phone
       raises its keyboard for -- comes to rest above the bar rather than under it. The
       margin clears the bar at its tallest, which is the cap above. */
    .designer #form input, .designer #form select,
    .designer #form textarea, .designer #form button { scroll-margin-bottom: 44vh; }
  }
  /* A phone with its keyboard up, or one on its side: the bar takes less of what little
     there is, and what it cannot show it scrolls. */
  @media (max-width: 900px) and (max-height: 560px) {
    .designer .footbar { max-height: 34vh; }
  }
  /* The opening position and the move map, side by side. Drawn on two boards because
     they are two claims: where the pieces start, and where one piece could go from an
     empty board. Overlaid, the dots and rings sat among pieces the map is not about. */
  .twoBoards { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); }
  .twoBoards figure { margin: 0; }
  /* Both boards are inputs: the opening board takes pieces, the lone piece's board
     takes the squares it should reach. */
  [data-board] svg { cursor: pointer; }
  .teachPanel { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--line); }
  /* The readings of a click, one sentence each. A step has four and a line has sixteen,
     and every one of them is a rule the engine confirmed -- so the list scrolls rather
     than being cut, and the board above shows whichever one is lit. */
  .teachOffers { list-style: none; margin: 8px 0 0; padding: 0; max-height: 236px;
                 overflow-y: auto; display: grid; gap: 4px; }
  .teachOffers button { display: block; width: 100%; text-align: left; font-size: 12.5px;
                        line-height: 1.35; padding: 5px 8px; min-height: var(--ctl-h-sm); }
  .palette { gap: 4px; margin-bottom: 4px; }
  .palette button { min-width: var(--ctl-h-sm); min-height: var(--ctl-h-sm); padding: 3px 5px; }
  .palette img { width: 24px; height: 24px; vertical-align: middle; }
  /* Every distinct piece the library holds. Fifty-odd rows, so the list scrolls inside
     the panel and each row is a name, what it does and where it is played. */
  .paletteList { list-style: none; margin: 8px 0 0; padding: 0; max-height: 300px;
                 overflow-y: auto; display: grid; gap: 4px; }
  .paletteList li { display: grid; grid-template-columns: auto 1fr auto; gap: 4px 8px;
                    align-items: baseline; }
  .paletteList button { justify-self: start; }
  .paletteGames { white-space: nowrap; }
  /* PF4: the one piece picker (V9's drawings and PF's ready pieces), a search and a chip a
     kind, then a grid of drawings, each on the board's dark square so a white piece reads.
     A press on a tile opens a line across the grid under its row (the drawing, what it is,
     how it moves, the buttons that take it), pushing the rows below down instead of lying
     over them. */
  .picker .chips { margin: 6px 0 4px; }
  .drawingGrid { list-style: none; margin: 6px 0 0; padding: 0; display: grid; gap: 6px;
                 grid-template-columns: repeat(auto-fill, minmax(78px, 1fr));
                 max-height: 340px; overflow-y: auto; }
  .drawingGrid > li:not(.pickAbout) { display: flex; }
  .drawingGrid > li > button { display: flex; flex-direction: column; align-items: center; gap: 3px;
                        flex: 1; min-width: 0; padding: 4px 2px; font-size: 11.5px;
                        line-height: 1.2; }
  .drawingGrid img { width: 40px; height: 40px; background: var(--sq-dk);
                     border-radius: var(--r-ctl); }
  .drawingGrid > li > button.worn { border-color: var(--act); box-shadow: inset 0 0 0 1px var(--act); }
  .drawingGrid > li > button[aria-expanded="true"] { border-color: var(--ink);
                                             box-shadow: inset 0 0 0 1px var(--ink); }
  .tileTag { font-size: 10.5px; color: var(--muted); font-weight: 400; }
  .pickAbout { grid-column: 1 / -1; display: flex; gap: 10px; align-items: flex-start;
               color: var(--ink); line-height: 1.45; border-left: 3px solid var(--note);
               padding: 4px 0 4px 9px; }
  .pickAbout > img { width: 56px; height: 56px; flex: none; background: var(--sq-dk);
                     border-radius: var(--r-ctl); }
  .pickAbout > div > div + div { margin-top: 3px; }
  .pickAbout .row { margin-top: 8px; gap: 6px; }
  .pickHead { font-weight: 600; margin-top: 14px; }
  /* PF4: a piece's card folds to one line, a real button: its drawing, its name and its
     moves in words, cut to the width; open, the line keeps the drawing and the name. */
  .pieceFold { display: flex; align-items: center; gap: 10px; width: 100%;
               min-height: var(--hit); padding: 0; border: 0; background: transparent;
               color: var(--ink); text-align: left; font-weight: 400; }
  .pieceFold:hover:not(:disabled) { border-color: transparent; }
  .pieceFold:hover .foldName { text-decoration: underline; }
  .foldMark { flex: none; width: 7px; height: 7px; margin: 0 3px 0 4px;
              border-right: 2px solid var(--muted); border-bottom: 2px solid var(--muted);
              transform: rotate(-45deg); }
  .pieceFold[aria-expanded="true"] .foldMark { transform: rotate(45deg); }
  .pieceFold .iconSlot { flex: none; }
  .foldName { font-weight: 600; white-space: nowrap; }
  /* Sized by the line, never by its words: a long sentence held to one line would
     otherwise widen the form past a phone's screen. */
  .foldSays { flex: 1 1 0; min-width: 0; contain: inline-size; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; }
  .editorPiece:not(.folded) .foldSays { display: none; }
  .pieceBody { margin-top: 10px; }
  .pieceBody[hidden] { display: none; }
  /* What each player holds off the board, under a still of the opening position. */
  .stillHand { gap: 5px; margin-top: 6px; align-items: baseline; }
  /* Q852: the line under a still board naming each wild piece's walk that jumps, with a
     swatch of the walk's own hue. */
  .walkLeaps { margin-top: 6px; color: var(--ink); }
  .walkLeaps > div { margin-top: 2px; }
  .walkName { display: inline-block; margin-right: 10px; white-space: nowrap; }
  .walkSwatch { display: inline-block; width: 10px; height: 10px; margin-right: 6px;
    border-radius: 50%; border: 1px solid var(--walk-edge); vertical-align: -1px; }
  .stillHand .chip { padding: 2px 8px; min-height: 0; }
  /* The design steps that apply to the draft, one sentence each. Thirty of them on an
     ordinary game, so the list scrolls inside the panel rather than pushing the rules in
     English off the column. */
  .tryList { margin: 8px 0 0; display: grid; gap: 4px; max-height: 300px;
             overflow-y: auto; }
  .tryList button { display: block; width: 100%; text-align: left; font-size: 12.5px;
                    line-height: 1.35; padding: 5px 8px; min-height: var(--ctl-h-sm); }
  /* The rules this draft cannot take, each with every reason, in plain words (Q995b). */
  .refusedList { list-style: none; margin: 8px 0 0; padding: 0; display: grid; gap: 6px; }
  .refusedList b { display: block; }
  .refusedReasons, .turnSaid ul { margin: 4px 0 0; padding-left: 18px; display: grid; gap: 4px; }
  /* How the players take turns (Q995b): the first panel's three ways, each a radio with its
     sentence, and under them what the last choice changed. */
  .turnWays { border: 0; margin: 0 0 10px; padding: 0; display: grid; gap: 6px; min-width: 0; }
  .turnWays > legend { font-size: 12px; color: var(--muted); padding: 0; margin-bottom: 4px; }
  .turnWay { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 8px;
             align-items: start; cursor: pointer; }
  .turnWay input { margin-top: 3px; }
  .turnWay .cardSays { display: block; }
  .turnSaid { margin: 4px 0 0; }
  .turnSaid button { margin-top: 6px; }
  /* Every piece rests (Q995d, fix 7): one sentence with its menu inside, beside *add a
     piece*, and under the row what every piece rests. */
  .restAll { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: 13px; }
  .restAll select { width: auto; min-height: var(--ctl-h-sm); padding: 3px 6px; }
  .restSentence { margin-top: 6px; }
  /* How a piece takes its turns (Q995c, fix 5): a row on the piece card, under its moves,
     separated by a rule rather than a border (no panel inside a panel). Each control keeps
     its label; what each does is one fold under the row. */
  .pieceTurns { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--line); }
  .pieceTurnsHead { color: var(--muted); margin-bottom: 4px; }
  .pieceTurnsRow { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: flex-start; }
  .pieceTurnsField { flex: 1 1 120px; min-width: 0; }
  /* A list of pieces (who it beats) takes the row's whole width, under the boxes. */
  .pieceTurnsField:has(.listControl) { flex-basis: 100%; order: 1; }
  .pieceTurnsField .field { margin-bottom: 4px; }
  .pieceTurnsSays ul { margin: 4px 0 0; padding-left: 16px; display: grid; gap: 4px; }
  .roundsLine { margin-top: 4px; }
  .roundsLine button.quiet, #dRuleFound button.quiet { padding: 0; min-height: 0;
    font-size: inherit; vertical-align: baseline; }
  .pieceSpeed { margin-top: 20px; white-space: nowrap; }
  /* A set speed in the tray: the number in a corner of the piece's button, in ink. */
  .palette button { position: relative; }
  .traySpeed { position: absolute; right: 2px; bottom: 0; font-size: 10px; line-height: 1;
               color: var(--ink); font-weight: 600; }
  /* Find a rule by typing (fix 6): the box above the menu, and what it found under it. */
  .ruleFind input[type="search"] { margin-bottom: 4px; }
  #dRuleFound { display: grid; gap: 4px; margin-bottom: 4px; }
  #dAddRuleSays .cardSays { margin-top: 0; }
  /* A save under a borrowed name asks for one of its own (fix 9). */
  .nameAsk { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--line); }
  @media (max-width: 720px) {
    .paletteList li { grid-template-columns: auto 1fr; }
    .paletteGames { grid-column: 2; }
  }
  .twoBoards figcaption { font-size: 11px; color: var(--muted); margin-top: 4px;
                          text-align: center; }
  /* Two hashes and what they mean, which is a table of three columns and not prose. */
  .identity { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px;
              font-size: 12.5px; align-items: baseline; }
  .identity code { font-size: 12px; }
  .field { display: grid; gap: 4px; margin-bottom: 10px; }
  .field > span { font-size: 12px; color: var(--muted); }
  .field input, .field select, .field textarea { width: 100%; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .editor { display: grid; gap: 10px; }
  /* Each piece is a sheet of its own on the form's ground (V8): the panel's paper, its
     radius and shadow, and the one being edited bordered in the action. */
  .editorPiece { background: var(--paper) 0 0 / 180px, var(--panel); border: 1px solid var(--line);
                 border-radius: var(--r-panel); padding: 12px; box-shadow: var(--shadow); }
  .editorPiece.on { border-color: var(--act); }
  .editorPiece .row { align-items: flex-start; }
  /* Eight controls, and the widths are the labels': "blockable leap" and "all eight ways"
     are the longest option in their columns, and a select that clips its own value is
     a control you have to open to read. */
  /* A row that wraps rather than a grid: the form column is narrower than eight
     controls on most screens, and a grid pushed the last of them under the preview.
     Each control carries its own label, so the widths key off what the control *is*
     rather than off where it sits -- a label added in the middle used to shift every
     width after it by one. */
  .moveRow { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: flex-end;
             margin-top: 8px; font-size: 13px; }
  .moveField { display: grid; gap: 2px; }
  .moveField > span { font-size: 10.5px; line-height: 1.2; color: var(--muted); }
  .moveRow input, .moveRow select { min-height: var(--ctl-h-sm); padding: 5px 6px; font-size: 12.5px;
                                    text-overflow: ellipsis; width: 100%; }
  .moveRow [title*="slides run"] { width: 104px; }
  .moveRow [title*="which turns"] { width: 118px; }
  .moveRow [title*="one step as"] { flex: 1 1 110px; min-width: 96px; }
  .moveRow [title*="how many squares"],
  .moveRow [title*="nearest square"],
  .moveRow [title*="only from this rank"] { width: 74px; }
  .moveRow [title*="a pawn moves"] { width: 108px; }
  .moveRow > button { align-self: flex-end; margin-bottom: 1px; }
  /* How far, how near and from which rank, folded under *more* until one says something
     (Q545): a whole line of its own, so the row above it stays the four controls. */
  .moveMore { flex: 1 1 100%; }
  .moveMore > summary { cursor: pointer; width: max-content; }
  .moveMoreBody { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: flex-end;
                  margin-top: 4px; }
  /* A piece's lines of sight (schema 30, Q999c): rows built like a way of moving's, under a
     head of their own, the widths keyed off the field each control writes. */
  .sightRows { margin-top: 10px; border-top: 1px dashed var(--line); padding-top: 6px; }
  .sightHead { line-height: 1.35; }
  .sightRow [data-sfor="effect"] { width: 104px; }
  .sightRow [data-sfor="whose"] { width: 136px; }
  .sightRow [data-sfor="vectors"] { flex: 1 1 110px; min-width: 96px; }
  .sightRow [data-sfor="symmetry"] { width: 118px; }
  .sightRow [data-sfor="through"] { width: 172px; }
  .sightRow [data-sfor="range"], .sightRow [data-sfor="min_range"] { width: 74px; }
  .sightAffects { display: grid; gap: 2px; flex: 1 1 100%; }
  .sightPicks { display: flex; flex-wrap: wrap; gap: 2px 12px; }
  .sightPick { display: inline-flex; align-items: center; gap: 4px; font-size: 12.5px; }
  .moveRow .sightPick input { width: auto; min-height: 0; padding: 0; }
  /* A command's `captures_own` box (Q1007d), beside *what it does*, level with the menus. */
  .sightRow .sightOwn { align-self: flex-end; min-height: var(--ctl-h-sm); max-width: 190px; }
  .sightError { margin-top: 4px; }
  /* Rule cards, generated from the schema. A sub-card is indented under its parent so
     a nested clause reads as belonging to the rule it is in. They sit inside the rules'
     panel, so each is a row apart by a rule, never a box inside it (V8, §3.4). */
  .ruleCard { padding: 9px 0; border-top: 1px solid var(--line); }
  .ruleCard:last-child { padding-bottom: 0; }
  .subCard { border-left: 2px solid var(--line); padding: 2px 0 2px 10px; margin: 6px 0 10px; }
  .cardHead { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .cardHead > span { color: var(--muted); font-size: 12px; }
  .cardDrop, .cardShow { margin-left: auto; min-height: var(--ctl-h-sm); min-width: var(--ctl-h-sm);
                        padding: 3px 9px; }
  .cardShow { font-size: 13px; line-height: 18px; }
  .cardShow + .cardDrop { margin-left: 0; }
  .cardSays { margin-top: 2px; line-height: 1.35; }
  .cardBody { display: grid; gap: 2px; margin-top: 4px; }
  .listControl { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .listControl input[data-cpart="text"] { flex: 1 1 140px; width: auto; }
  .mapRow { display: flex; gap: 8px; align-items: baseline; }
  .mapRow .field { flex: 1; margin-bottom: 4px; }
  .eitherHead { gap: 6px; align-items: center; }
  .eitherHead > span { font-size: 12px; color: var(--muted); }
  details.extras { margin-top: 6px; }
  details.extras .cardBody { padding: 6px 0 0 10px; }
  .iconSlot { min-width: 40px; height: 40px; border: 1px dashed var(--line); border-radius: var(--r-ctl);
              display: flex; align-items: center; justify-content: center; gap: 2px;
              padding: 0 3px; overflow: hidden; }
  .iconSlot img { width: 34px; height: 34px; }
  /* Both seats, where a piece is drawn its own way for each. */
  .iconSlot img.half { width: 28px; height: 28px; }
  .sliders { display: grid; gap: 7px; }
  .slider { display: grid; grid-template-columns: 150px 1fr 52px; gap: 9px;
            align-items: center; font-size: 13px; }
  .slider input[type=range] { width: 100%; padding: 0; border: none; background: none; }
  .slider .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  @media (max-width: 620px) { .slider { grid-template-columns: 1fr 44px; }
                              .slider input[type=range] { grid-column: 1 / -1; } }
  .ok { border: 1px solid var(--good); color: var(--good); border-radius: var(--r-ctl);
        padding: 9px 12px; margin: 12px 0; }

  /* --- rank and file labels, inside the edge squares of a board that wraps --- */
  svg#board text.co { font-family: var(--f-mono); font-weight: 600; opacity: .75; pointer-events: none;
                      font-variant-numeric: tabular-nums; user-select: none; }

  /* --- the library as a map ---
     The frame, the controls and the panel beside them. The drawing inside the frame is a
     canvas the graph library owns, and every colour, width and dash in it is written by
     `page/explorer.js` in the library's own stylesheet -- so what is here is the HTML the
     canvas sits among. */
  .mapView { display: grid; gap: 16px; align-items: start;
             grid-template-columns: minmax(0, 1fr) minmax(260px, 320px); }
  .mapFrame { position: relative; overflow: hidden; background: var(--panel);
              border: 1px solid var(--line); border-radius: var(--r-panel);
              box-shadow: var(--shadow); height: min(74vh, 780px); }
  /* The drawing fills the frame. The graph library measures the element it is given, so
     it needs one with a size of its own; and a gesture on it is the library's, never the
     page's scroll. */
  .exploreCanvas { position: absolute; inset: 0; touch-action: none; }
  /* Every game on the map as a button, because a drawing on a canvas has no Tab
     stops of its own. Seventy visible buttons would be a second map, so each is clipped
     where it sits and the focused one shows itself in the frame's corner. Clipped rather
     than hidden: `display: none` and `visibility: hidden` both take an element out of the
     Tab order, which is the one thing these are for. */
  .exploreKeys button { position: absolute; left: 8px; top: 8px; z-index: 2;
                        max-width: calc(100% - 16px); text-align: left;
                        background: var(--panel); border-color: var(--act);
                        clip-path: inset(50%); }
  .exploreKeys button:focus { clip-path: none; }
  /* The local view's bar, over the drawing's top edge: what it shows and the ways out. */
  .mapLocal { position: absolute; left: 9px; right: 9px; top: 9px; z-index: 1;
              display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center;
              padding: 6px 10px; background: var(--panel); border: 1px solid var(--line);
              border-radius: var(--r-ctl); box-shadow: var(--shadow-pop); }
  .mapLocal button { min-height: var(--ctl-h-sm); padding: 3px 10px; }
  /* A control the map hides until it applies: the arrows' key, the score key, the local
     view's bar. Each carries a `display` of its own, which would beat `hidden`. */
  .mapLegend [hidden], .mapFrame [hidden] { display: none !important; }
  /* A community's name over its outline, on the outlines' own SVG layer under the dots.
     Ink with a halo of the page behind it, as a dot's name is drawn; the colour is the
     outline's. It answers no pointer: a press goes through to the drawing. */
  .mapHullName { fill: var(--ink); stroke: var(--panel); stroke-linejoin: round;
                 paint-order: stroke; font-weight: 600; pointer-events: none; }
  .mapHullName.faded { fill: var(--faint); }
  .mapTools { position: absolute; right: 9px; bottom: 9px; display: flex; gap: 4px; }
  .mapTools button { min-height: var(--ctl-h-sm); padding: 3px 10px; background: var(--panel); }
  .chip .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
               margin-right: 6px; vertical-align: -1px; }
  .chip .swatch { margin-right: 6px; vertical-align: -1px; }
  .chip.off { opacity: .45; text-decoration: line-through; }
  .mapLegend { display: grid; gap: 7px; margin-bottom: 12px; }
  /* The map's controls (V8): one sheet above the drawing, its rows apart by space, never
     a panel inside it. On a phone it folds to its summary line (`wireExplorer`), so the
     drawing is on the first screen. */
  .mapControls { margin-bottom: 16px; padding: 10px 15px; }
  .mapControls > summary { min-height: var(--hit); display: flex; align-items: center;
                           width: max-content; max-width: 100%; }
  .mapControls[open] > summary { margin-bottom: 4px; }
  .mapCtl { gap: 8px 16px; margin: 8px 0; }
  .mapEnds { justify-content: space-between; }
  /* The size row's sentence takes the room between its menu and *reset*, which ends the
     row; the communities are listed once, as the key, and fold from their panel (Q801). */
  .mapSay { flex: 1 1 200px; min-width: 0; }
  .mapControls .mapLegend { margin-bottom: 4px; }
  .mapPanel .links { margin-top: 10px; max-height: 42vh; overflow-y: auto; }
  .mapPanel .linkRow { display: block; width: 100%; text-align: left; border: none;
                       min-height: var(--hit); background: none; padding: 6px 0; border-top: 1px solid var(--line);
                       color: var(--ink); }
  .mapPanel .linkRow:hover { background: var(--sunk); }
  @media (max-width: 900px) {
    .mapView { grid-template-columns: 1fr; }
    /* Short enough that there is page left above and below to scroll by: a finger on the
       map itself moves the map. */
    .mapFrame { height: 56vh; }
  }
  @media (min-width: 901px) {
    .mapPanel { position: sticky; top: 12px; max-height: calc(100vh - 24px);
                overflow-y: auto; }
  }

  /* --- what's new, the changelog --- */
  /* A column of prose, so it is capped at a readable measure rather than at the page's
     width, and the cap is the only thing a phone has to undo. */
  /* V7: the page's title at the title size; each day a sheet of the panel's paper with its
     entries ruled inside it, the date beside the sheet at a desk (held in view while its
     day scrolls by) and above it on a phone. */
  .news { max-width: 880px; }
  .news > h1, .news > h2 { font-size: 28px; line-height: 34px; margin: 0 0 6px; }
  .newsFeeds { margin: 8px 0 0; }
  .news > .small { max-width: 680px; margin-bottom: 4px; }
  .newsDay { margin-top: 26px; display: grid; gap: 8px 24px; }
  .newsDay > h3 { margin: 0; font-size: 17px; line-height: 24px; }
  .newsSheet { padding: 4px 20px 16px; }
  @media (min-width: 900px) {
    .newsDay { grid-template-columns: 150px minmax(0, 1fr); }
    .newsDay > h3 { position: sticky; top: 12px; align-self: start; padding-top: 14px; }
  }
  .newsEntry { padding-top: 14px; margin-top: 14px; }
  .newsEntry + .newsEntry { border-top: 1px solid var(--line); }
  .newsEntry:first-child { margin-top: 0; }
  .newsEntry h4 { font-size: 16px; line-height: 22px; margin: 0 0 6px; }
  .newsEntry p { margin: 0 0 9px; }
  .newsEntry p:last-child { margin-bottom: 0; }
  .newsEntry code { font: 12.5px/1.45 var(--f-mono);
                    background: var(--sunk); border-radius: 5px; padding: 1px 5px;
                    overflow-wrap: anywhere; }
  .newsEntry hr { border: none; border-top: 1px solid var(--line); margin: 12px 0; }

  /* --- tabs, on the board page --- */
  .tabs { margin-bottom: 10px; }
  .tabpane { max-height: 58vh; overflow-y: auto; }"""

# LP4 (`docs/plan-landing.md` §3.2, direction B, the mock-up's `b.css` on the site's tokens):
# the question and its turning endings beside the game each became; then one idea, start to
# finish, the board pinned at 1400 beside six steps and a board in steps 2 and 4 on a phone;
# the three beats (the failure played before its verdict, the card's words struck and
# replaced, the fix played before its verdict and the paired qualities). The board's motion
# is `motion.js`'s; what moves here is CSS (a turning clause, a struck word, a growing bar),
# and under reduced motion nothing does.
RULES += """
  .landingB .heroB { grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); gap: 48px;
                     margin: 16px 0 40px; }
  .landingB .heroB .landingTitle { font-size: 46px; line-height: 52px; letter-spacing: -.01em;
                         margin: 0 0 14px; }
  .landingB .heroB .wcbi { display: block; color: var(--muted); }
  .landingB .heroB .clauses { display: grid; min-height: 156px; }
  .landingB .heroB .clause { grid-area: 1 / 1; align-self: start; opacity: 0; transform: translateY(24px);
    transition: opacity .55s ease, transform .55s cubic-bezier(.2, .8, .2, 1);
    text-decoration: underline; text-decoration-color: var(--family);
    text-decoration-thickness: 5px; text-underline-offset: 8px; text-decoration-skip-ink: none; }
  .landingB .heroB .clause.on { opacity: 1; transform: none; }
  .landingB .heroB .clause.off { opacity: 0; transform: translateY(-24px); }
  .landingB .heroB .landingLead { font-size: 19px; line-height: 28px; max-width: 34em; }
  .landingB .heroB .landingWays a.button { min-height: var(--ctl-h-lg); padding: 10px 20px; font-size: 17px; }
  .landingB .heroShow { margin: 0; min-width: 0; }
  .landingB .heroStage { display: grid; grid-template-columns: minmax(0, 1fr); }
  .landingB .heroState { grid-area: 1 / 1; opacity: 0; transform: scale(.97) rotate(-.6deg);
               pointer-events: none; transition: opacity .7s ease, transform .7s ease; }
  .landingB .heroState.on { opacity: 1; transform: none; pointer-events: auto; }
  .landingB .heroBoard { max-width: 440px; }
  .landingB .heroBoard:empty { aspect-ratio: 1; }
  .landingB .heroState figcaption { display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
                          margin-top: 10px; }
  .landingB .heroState figcaption a { font-weight: 600; }
  .landingB .familyChip { display: inline-flex; align-items: center; gap: 6px; }
  .landingB .familyChip .swatch, .landingB .ruleCard .swatch { width: 10px; height: 10px; border-radius: 3px;
                                           flex: none; background: var(--family); }
  .landingB .heroDots { margin-top: 8px; gap: 6px; align-items: center; }
  .landingB .heroDot { width: 12px; height: 12px; min-height: 0; padding: 0; border-radius: 50%;
             border: 1px solid var(--edge); background: transparent; }
  .landingB .heroDot.on { background: var(--ink); border-color: var(--ink); }
  .landingB .journey { margin: 8px 0 40px; }
  .landingB .journeyTitle { font-size: 30px; line-height: 36px; margin: 0 0 16px; }
  .landingB .journeyGrid { display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 48px; }
  .landingB .pin { position: sticky; top: 16px; align-self: start; margin: 0; }
  .landingB .pinStatus { min-height: 40px; display: flex; align-items: center; font-weight: 600; }
  .landingB .pinTray, .landingB .beatTray { gap: 10px; align-items: center; min-height: 26px; margin-top: 8px; }
  .landingB .pinVerdict { margin-top: 10px; min-height: 44px; border-radius: 8px; padding: 10px 12px;
                background: var(--sunk); opacity: 0; transform: translateY(6px);
                transition: opacity .4s, transform .4s; }
  .landingB .pinVerdict.shown { opacity: 1; transform: none; }
  .landingB .pinVerdict.bad { color: var(--bad); }
  .landingB .pinVerdict.good { color: var(--good); }
  .landingB .beats { list-style: none; margin: 0; padding: 0; }
  .landingB .beat { min-height: 60vh; padding: 20px 0 20px 24px; border-left: 3px solid var(--line);
          opacity: .4; transition: opacity .5s, border-color .5s; }
  .landingB .beat.active, .landingB .beat.seenAll { opacity: 1; border-left-color: var(--act); }
  .landingB .beat h3 { font-size: 25px; line-height: 32px; margin: 6px 0 12px; max-width: 24em; }
  .landingB .beat p { margin: 10px 0 0; max-width: 40em; }
  .landingB .beatNo { color: var(--muted); margin: 0; font-size: 14px; }
  .landingB .beatBoard, .landingB .beatTray { display: none; }
  .landingB .say { position: relative; padding-left: 30px; }
  .landingB .say::before { position: absolute; left: 0; top: 0; font-family: var(--f-mono); }
  .landingB .say.bad::before { content: "\\2717"; color: var(--bad); }
  .landingB .say.good::before { content: "\\2713"; color: var(--good); }
  .landingB .say.change::before { content: "\\2192"; color: var(--ink); }
  .landingB .ruleCard { display: flex; gap: 10px; align-items: flex-start; border: 1px solid var(--line);
              border-left: 4px solid var(--family); border-radius: 8px; padding: 8px 10px;
              background: var(--panel); max-width: 30em; margin-top: 10px; }
  .landingB .ruleCard .swatch { margin-top: 6px; }
  .landingB .ruleCard > div { display: flex; flex-direction: column; }
  .landingB .ruleCode { margin: 10px 0 0; padding: 8px 10px; border-radius: 8px; background: var(--sunk);
              overflow-x: auto; font: 13px/19px var(--f-mono); max-width: 40em; }
  .landingB .ruleCode.diff del { color: var(--bad); }
  .landingB .ruleCode.diff ins { color: var(--good); text-decoration: none; font-weight: 600; }
  .landingB .ruleCard.edit del, .landingB .swap del { text-decoration: none; color: var(--muted);
    background: linear-gradient(var(--bad), var(--bad)) no-repeat 0 58% / 0 2px;
    transition: background-size .7s ease .3s, color .7s ease .3s; }
  .landingB .ruleCard.edit ins, .landingB .swap ins { text-decoration: none; font-weight: 600;
    color: var(--good); display: inline-block; opacity: 0; transform: translateY(6px);
    transition: opacity .45s ease 1.05s, transform .45s ease 1.05s; }
  .landingB .beat.seen .ruleCard.edit del, .landingB .beat.seenAll .ruleCard.edit del, .landingB .swap.go del { background-size: 100% 2px; }
  .landingB .beat.seen .ruleCard.edit ins, .landingB .beat.seenAll .ruleCard.edit ins, .landingB .swap.go ins { opacity: 1; transform: none; }
  .landingB .playWords { margin-top: 10px; color: var(--muted); min-height: 20px; }
  .landingB .playCtl { margin-top: 6px; gap: 6px; }
  .landingB .playCtl button { min-width: 44px; }
  .landingB .verdictBlock { transition: opacity .5s, transform .5s; }
  .landingB .verdictBlock.pending { opacity: 0; transform: translateY(8px); visibility: hidden; }
  .landingB .out { border-left: 3px solid var(--bad); padding-left: 10px; }
  .landingB .out.good { border-left-color: var(--good); }
  .landingB .verdict-word-bad { color: var(--bad); }
  .landingB .verdict-word-good { color: var(--good); }
  .landingB .exampleNote { border-left: 3px solid var(--act); padding-left: 10px; }
  .landingB .quickSaid { margin: 8px 0 0; padding-left: 20px; }
  .landingB .pairBars { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 6px 12px;
              align-items: center; margin-top: 14px; max-width: 30em; font-size: 14px; }
  .landingB .pairBar { position: relative; height: 10px; }
  .landingB .pairBar b, .landingB .pairBar i { position: absolute; left: 0; right: 0; transform-origin: left;
    --v: 0; transform: scaleX(var(--v)); transition: transform .9s cubic-bezier(.2, .7, .2, 1);
    border-radius: 3px; }
  .landingB .pairBar b { top: 0; bottom: 0; background: var(--meter); opacity: .3; }
  .landingB .pairBar i { top: 3px; bottom: 3px; background: var(--meter); }
  .landingB .pairBar .low { background: var(--bad); }
  .landingB .verdictBlock.pending .pairBar b, .landingB .verdictBlock.pending .pairBar i { transform: scaleX(0); }
  .landingB .pairKey .key { display: inline-block; width: 16px; height: 6px; border-radius: 3px;
                  background: var(--meter); margin: 0 6px 0 12px; }
  .landingB .pairKey .key.faint { opacity: .3; margin-left: 0; }
  .landingB .ladder { display: flex; flex-wrap: wrap; gap: 6px; min-height: 34px; }
  .landingB .ladder span { border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px;
                 background: var(--panel); }
  .landingB .ladder i { font-style: normal; color: var(--muted); font-size: 13px; margin-left: 4px; }
  .landingB .challengeLink { background: var(--sunk); border-radius: 8px; padding: 8px 10px; max-width: 24em; }
  .landingB .handCount { display: inline-flex; align-items: center; gap: 4px; font-weight: 600; }
  .landingB .swap .handCount, .landingB .swap .muted { margin-left: 6px; }
  .landingB .claimKey { display: inline-block; width: 12px; height: 12px; border-radius: 3px; }
  .landingB .claimKey.p0 { background: var(--claim0); }
  .landingB .claimKey.p1 { background: var(--claim1); }
  .landingB .otherStories { margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--line); }
  .landingB .otherStories p { margin: 0; }
  .landingB .otherStories .row { gap: 8px; flex-wrap: wrap; margin-top: 8px; }  @media (max-width: 900px) {
    .landingB .heroB { grid-template-columns: minmax(0, 1fr); gap: 16px; margin: 4px 0 28px; }
    .landingB .heroB .landingTitle { font-size: 30px; line-height: 36px; }
    .landingB .heroB .clauses { min-height: 108px; }
    .landingB .heroB .clause { text-decoration-thickness: 3px; text-underline-offset: 5px; }
    .landingB .heroB .landingLead { font-size: 17px; line-height: 25px; }
    .landingB .heroB .landingWays a.button { padding: 8px 16px; font-size: 16px; }
    .landingB .heroShow { max-width: 360px; }
    .landingB .journeyGrid { grid-template-columns: minmax(0, 1fr); gap: 0; }
    .landingB .pin { display: none; }
    .landingB .beat { min-height: 0; opacity: 1; padding: 14px 0 18px 14px; border-left-color: var(--act); }
    .landingB .beat h3 { font-size: 21px; line-height: 27px; }
    .landingB .beatBoard { display: block; max-width: 300px; margin-top: 12px; }
    .landingB .beatTray { display: flex; flex-wrap: wrap; }
    .landingB .journeyTitle { font-size: 24px; line-height: 30px; }
    .landingB .otherStories { display: flex; align-items: center; gap: 10px; }
    .landingB .otherStories p { flex: none; max-width: 40%; }
    .landingB .otherStories .row { flex-wrap: nowrap; overflow-x: auto; margin-top: 0; min-width: 0; }
    .landingB .otherStories .chip { flex: none; }
  }
  /* Reduced motion: nothing turns, plays, grows or is pinned; each game's board sits in its
     own step, stepped by press, and every edit and verdict shows at once. */
  @media (prefers-reduced-motion: reduce) {
    .landingB *, .landingB *::before, .landingB *::after {
      transition: none !important; animation: none !important; }
    .landingB .pin { display: none; }
    .landingB .journeyGrid { grid-template-columns: minmax(0, 1fr); }
    .landingB .beat { min-height: 0; opacity: 1; border-left-color: var(--act); }
    .landingB .beatBoard { display: block; max-width: 360px; margin-top: 12px; }
    .landingB .beatTray { display: flex; flex-wrap: wrap; }
  }
  .landingB .beatTray:empty, .landingB .pinTray:empty { display: none; }"""

# LP2: each family of rules on the landing's wall of words reads its colour from its own
# class (choice E1 of `docs/plan-landing.md` §4: the map's per-kind hues), made from
# `landing.BLOCKS` so a row and its colour are named in one place.
RULES += "".join(f"\n  .{block.hue} {{ --family: var(--{block.colour}); }}"
                 for block in BLOCKS)

# The three families, each with its system fallback after it (§3.1).
FAMILIES = f"""
  :root {{ --f-head: {HEAD}; --f-body: {BODY}; --f-mono: {MONO}; }}"""

STYLE = TOKENS + FAMILIES + faces() + RULES
