// The library as a graph somebody steers, drawn by Cytoscape.js.
//
// `docs/archive/plan-game-graph.md` §8, stages G8 to G13. The library's map, at `#/map`,
// drawn from one call to `GET /api/graph`. `page/graph.js` holds the legend, the panels and
// every reading of the payload both modules share; what is here is the drawing itself and
// the gestures over it.
//
// **It places and never computes.** Every coordinate the map opens on is the server's, in
// layout units of one node radius, and this module multiplies each by one constant to get
// the library's own model pixels. After that a position is a reader's -- a drag, or a
// *reset* back to the server's -- and nothing else here writes one. There is no layout on
// the page: the server's picture keeps games alike near each other and remembers itself
// from one build to the next (`docs/plan-map.md` M3), and one here would only draw another
// (*tidy*, retired in Q477). It works out no distance, compares no rules, reads no spec,
// formats no number and asks the engine nothing. What is decided here is presentation: how
// big a dot is drawn, what colour a line is, what a tap holds up.
//
// Three things the graph library needs that an SVG did not:
//
// **Cytoscape cannot read a custom property.** `var(--ink)` means nothing to a canvas, so
// every colour is resolved first -- through one off-screen probe element, which is also what
// turns `mapScoreFill`'s `color-mix` into the `rgb()` the library will take. One function,
// and the page's own ramp stays the only place a score becomes a colour. A theme change is
// therefore a repaint, because a resolved colour is the colour of the theme that was on when
// it was read.
//
// **A dot cannot carry two rings.** A node has one border, and a game the checks ruled out
// needs a second mark inside its own radius. So the ring is a node of its own at the same
// place, unclickable and never dragged. Whatever moves a dot has to move its ring, and
// `explorerRingsFollow` is the one place that knows it -- a drag and a *reset* both end in
// a position, and a position is what it watches.
//
// **A community folds into one dot.** The payload's `communities` are a partition of the
// library, so a game folds into one place or none, and the lines its games had to the rest
// of the library are drawn once from the folded dot with a count on them. Nothing moves when
// a community folds, which is why opening it again puts every game back where it was.
// A drag is the one thing that moves a dot, and *reset* is the way back to the server's
// picture.
//
// **A community can be outlined.** A switch draws a smooth outline round each community of
// three or more (`docs/archive/plan-graph-map.md` S5), in its colour and under the server's
// name for it -- one outline for each place the server says it is drawn (`parts`), so a
// community drawn in two places is two outlines in one colour rather than one stretched
// across the map. The outline is the vendored `cytoscape-bubblesets`, which draws on an SVG
// layer under the dots from where the drawn dots and lines are -- so the outline is pixels,
// like a line's width, and never a fact: which games are in a community is the server's.
// Whatever changes what is drawn -- a fold, the local view, *hide the rest* -- draws the
// outlines again over what is left.
//
// **More than one thing can decide how big a dot is.** The payload carries a menu of size
// keys, each with its rule in one sentence and its two real ends, and every dot carries its
// place between those ends as a number from 0 to 1. Switching keys resizes dots and moves
// none. A key a dot has no value under -- a game with no current score, under *score* --
// draws at the smallest size, and the sentence under the menu says so.
//
// **The panels, the chips, the legend and the find box are `page/graph.js`'s.** They are
// HTML beside the drawing and read the payload and `MAP_PICK`, so this module calls them
// rather than writing a second opinion about the same library. What is this module's is the
// paint -- a class on a Cytoscape element, and a width or a colour in the library's own
// stylesheet.
//
// **A filter lights; it never hides** -- with one exception a reader asks for by name.
// *Hide the rest* takes the unlit games off the drawing, and a line with an end among them
// goes with it, which is the library's own rule for an edge whose end is not displayed. The
// sentence under the controls says which of the two is happening, because *faded rather than
// gone* is untrue once they are gone.

// ---------------------------------------------------------------------- the explorer

let EXPLORER = null,       // the Cytoscape instance, or null when nothing is drawn
    EXPLORER_LOADING = null,  // the one request for the library, shared by every caller
    EXPLORER_PROBE = null,    // the element a CSS colour is resolved through
    EXPLORER_INK = null,      // the one pixel a resolved colour is read off
    EXPLORER_ONLY = false,    // *hide the rest*: the unlit games are taken off the drawing
    // Which of the payload's size keys is drawing the dots. The first the payload offers
    // when the page opens, which is the size the map has always drawn.
    EXPLORER_SIZE = '',
    // The site's own numbers for these same games, or null where there is no store to
    // read them out of and where the request failed. Fetched once per visit.
    EXPLORER_PEOPLE = null,
    // Which of the overlay's filters is narrowing the map, by the overlay's own name.
    // Empty is every game, which is what the page opens on.
    EXPLORER_KEEP = '',
    // The groups drawn as one dot, by the payload's own id. Nothing is folded when the
    // page opens, and *reset* empties it.
    EXPLORER_FOLDED = new Set(),
    // The local view: {at, steps}, one game by its index and how many steps out from it
    // are shown, or null for the whole map.
    EXPLORER_LOCAL = null,
    // The hashes a pointer resting on a game lights -- it and its neighbours -- or null.
    EXPLORER_NEAR = null,
    // The find box's first match, by hash: centred, marked and named first.
    EXPLORER_FIRST = null,
    // The one request for the outlines' two files, shared by every caller.
    EXPLORER_HULL_LOADING = null,
    // Whether the communities are outlined, the outline library on this drawing, and each
    // outline drawn, by community id: {path, key, label, members, colour}.
    EXPLORER_HULLS = false,
    EXPLORER_BUBBLES = null,
    EXPLORER_OUTLINES = new Map();

// The vendored files, by the names the manifest gives them. The version is in each name,
// so the route answers `immutable` and a new version is a new request rather than new
// bytes under the old one. The graph library comes first; the outlines' two follow it and
// in this order, because the second reads the first's global as it loads and both
// register with the library they find.
const EXPLORER_VENDOR = '/static/vendor/';
const EXPLORER_SRC = 'cytoscape-3.34.3.min.js';
const EXPLORER_HULL_SRC = ['cytoscape-layers-3.1.0.min.js', 'cytoscape-bubblesets-4.1.0.min.js'];
// Model pixels per layout unit. The payload's coordinates are in units of one node
// radius; the library wants a number it can draw a line width and a border against, and
// this is the one place the two are tied together.
const EXPLORER_UNIT = 34;
// A folded group's dot, across, in layout units. Wider than the widest game, because it
// stands for several of them and carries a name that has to be read at the whole-map view.
const EXPLORER_GROUP = 2.6 * EXPLORER_UNIT;
// The radius a dot with the smallest value draws at, in layout units, and the radius the
// largest draws at. G3's range, and every size key runs over it: the layout keeps one
// diameter between two dots, so a full-sized dot is the largest that cannot touch its
// neighbour whichever data is driving it.
const EXPLORER_SMALLEST = 0.5;
const EXPLORER_LARGEST = 1.0;
// Air round the drawing when it is fitted, in screen pixels, how far one press of the
// zoom buttons goes, and how far one arrow key moves the picture.
const EXPLORER_AIR = 26;
const EXPLORER_STEP = 1.4;
const EXPLORER_PAN = 90;
// A name is drawn at this many pixels whatever the zoom, so its size in model units is
// worked out again on every view change. The second number is the width of an average
// character at that size, which is how the greedy label pass measures a name, and the
// third is the air round it, so two names that only just miss are not read as one word.
const EXPLORER_TEXT_PX = 12.5;
const EXPLORER_TEXT_WIDTH = 0.58;
const EXPLORER_TEXT_AIR = 0.6;
// A line's width, in model pixels, mapped straight from the server's `likeness`: 0 is two
// games sharing nothing rare and 1 one game written two ways, and the width runs from a
// hair the eye still finds to the widest a line can be beside a dot.
const EXPLORER_WIDTH = 'mapData(likeness, 0, 1, 1.5, 12)';
// The least zoom the find box's first match is shown at, so a match at the whole-map
// view comes up close enough to read its neighbours.
const EXPLORER_HOLD_ZOOM = 0.8;
// The most steps out the local view goes. Two steps from the busiest game is already
// most of its community and the next one's.
const EXPLORER_STEPS_OUT = 2;
// Whether the map opens with the communities outlined.
const EXPLORER_HULLS_OPEN = true;
// How far an outline reaches round what it holds, in layout units: full strength within
// the first number of a dot and nothing past the second, and the same for a line. Every
// drawn game outside the community is handed to the outline as one to keep clear of, so
// an outline bends round a game that is not its own rather than taking it in.
const EXPLORER_HULL_REACH = {
  nodeR0: 0.9 * EXPLORER_UNIT, nodeR1: 1.9 * EXPLORER_UNIT,
  edgeR0: 0.45 * EXPLORER_UNIT, edgeR1: 1.0 * EXPLORER_UNIT,
  morphBuffer: 0.5 * EXPLORER_UNIT, pixelGroup: 5, virtualEdges: true,
};
// The fewest games an outline is drawn round. One dot outlined alone is a ring, and says
// less than the dot's own colour does.
const EXPLORER_HULL_LEAST = 2;
// A community's name wraps onto a new line past about this many characters.
const EXPLORER_HULL_LINE = 26;
// How long after a *reset* every outline is worked out again, in milliseconds: past the
// outline library's own wait between two redraws of one outline (its `throttle`, 100).
const EXPLORER_HULL_SETTLE = 250;

async function showExplorer() {
  stopPolling();
  SESSION = null;
  SHOWN = null;
  MAP_PICK = null;
  MAP_ON = null;
  MAP_FIND = '';
  MAP_SHOW = '';
  EXPLORER = null;
  EXPLORER_ONLY = false;
  EXPLORER_FOLDED = new Set();
  EXPLORER_SIZE = '';
  EXPLORER_PEOPLE = null;
  EXPLORER_KEEP = '';
  EXPLORER_LOCAL = null;
  EXPLORER_NEAR = null;
  EXPLORER_FIRST = null;
  EXPLORER_HULLS = EXPLORER_HULLS_OPEN;
  EXPLORER_BUBBLES = null;
  EXPLORER_OUTLINES = new Map();
  MAP_PAINT = 'community';
  // The picture the explorer opens on: the lines of likeness, and the arrows off.
  MAP_ARROWS = false;
  view.innerHTML = `
    <div class="row u-jc-space-between u-mb-12">
      <div class="muted small">Every game in the library, each drawn near the games most
        like it. Drag to move, scroll or pinch to zoom, tap a game or a line.</div>
      <div class="row">
        <input type="search" id="mapFind" placeholder="find a game"
          aria-label="find a game on the map" class="u-maxw-190">
        <label class="pick">light up
          <select id="mapShow" aria-label="which games to light up">
            <option value="">every game</option>
            <option value="yes">the ones with a score</option>
            <option value="viable">scored, and not ruled out</option>
            <option value="stale">scores out of date</option>
            <option value="no">never measured</option>
          </select></label>
        <label class="pick"><input type="checkbox" id="exploreOnly"> hide the rest</label>
        ${viewSwitchHtml('map')}
      </div>
    </div>
    <div id="exploreSay" class="muted small u-m-n6-0-10"></div>
    <div id="mapLit" class="muted small u-m-n6-0-10"></div>
    <div id="exploreBody"></div>`;
  wireViewSwitch();
  // The first call on a server with no map of this library reads every game and draws the
  // picture afresh, which is several seconds; every call after it is quick, because the map
  // is remembered. So the wait says which of the two this is rather than spinning silently.
  pending('Drawing the map. The first time, every game is read and placed.');
  // The first call can take seconds, and a person may leave the map before it answers.
  // Every await below is followed by `left()`, so a late answer draws nothing on the view
  // that replaced this one -- neither the body nor the header's tagline.
  const body = document.getElementById('exploreBody');
  const left = () => document.getElementById('exploreBody') !== body;
  const data = await api('/api/graph');
  if (left()) return;
  MAP_DATA = data;
  MAP_AT = {};
  MAP_DATA.nodes.forEach(node => { MAP_AT[node.hash] = node; });
  EXPLORER_PEOPLE = await explorerPeopleOverlay();
  if (left()) return;
  // The payload's first size key, which is the size the map has always drawn.
  EXPLORER_SIZE = explorerSizes().length ? explorerSizes()[0].key : '';
  clearPending();
  document.getElementById('tagline').textContent = mapSummary();
  body.innerHTML = explorerBodyHtml();
  const ready = await explorerLibrary();
  if (left()) return;
  if (!ready) { explorerSayBlocked(); return; }
  explorerDraw();
  wireExplorer();
  explorerPaintPanel();
  explorerSetHulls(EXPLORER_HULLS);
}

// The legend, the frame and the panel. The legend is `page/graph.js`'s, chips and all:
// the communities, what a line means, and both ends of the score ramp as numbers.
function explorerBodyHtml() {
  return `<details class="panel mapControls" id="mapControls" open>
    <summary class="muted small">what the map draws, and its key</summary>
    ${explorerLinesHtml()}
    ${mapLegendHtml()}
    ${explorerSizeHtml()}
    ${explorerKeepHtml()}
    </details>
    <div class="mapView">
      <div class="mapFrame" id="exploreFrame">
        <div class="exploreCanvas" id="explore"></div>
        <div class="mapLocal" id="exploreLocal" hidden></div>
        ${explorerKeysHtml()}
        <div class="mapTools">
          <button id="exploreOut" aria-label="zoom out">&minus;</button>
          <button id="exploreIn" aria-label="zoom in">+</button>
          <button id="exploreHome">whole map</button>
        </div>
      </div>
      <div class="panel mapPanel" id="mapPanel"></div>
    </div>`;
}

// ------------------------------------------------------------- the lines, and arrows
//
// `docs/plan-map.md` M4. The map's lines are the payload's `neighbours`, every one of
// them drawn: there is no cut to choose, because which games are alike enough for a line
// is each game's own nearest, not one threshold over the library. The arrows from a game
// to one made from it are one switch, off when the map opens, and a dot's fill is the
// community it is drawn in or its score, whichever the reader asks for.

function explorerLinesHtml() {
  return `<div class="row mapCtl">
    <label class="pick">colour the dots by
      <select id="explorePaint" aria-label="what the colour of a dot says">
        <option value="community"${MAP_PAINT === 'community' ? ' selected' : ''}>community</option>
        <option value="ramp"${MAP_PAINT === 'ramp' ? ' selected' : ''}>score</option>
      </select></label>
    ${mapGroups().length ? `<label class="pick" title="A soft outline round each community
      of three or more games, in its colour and under its name."><input type="checkbox"
      id="exploreHulls"${EXPLORER_HULLS ? ' checked' : ''}> outline the communities</label>`
      : ''}
    ${MAP_DATA.lineage.length ? `<label class="pick" title="An arrow from a game to one
      made as a new version of it, drawn however far apart the two are."><input
      type="checkbox" id="exploreArrows"${MAP_ARROWS ? ' checked' : ''}> arrows to the
      games made from each</label>` : ''}
  </div>`;
}

// The arrows on or off, with their key. What a fold carries, what a pick lights and what
// the panel lists follow from it, so it is one repaint of each.
function explorerSetArrows(on) {
  MAP_ARROWS = !!on;
  MAP_ON = mapLit();
  const box = document.getElementById('exploreArrows');
  if (box) box.checked = MAP_ARROWS;
  const key = document.getElementById('mapArrowsKey');
  if (key) key.hidden = !MAP_ARROWS;
  explorerPaintLines();
  explorerFoldPaint();
  explorerPaintLit();
  explorerPaintPanel();
}

// Community or score. The score ramp's key shows only while the fill is the score.
function explorerSetPaint(paint) {
  MAP_PAINT = paint === 'ramp' ? 'ramp' : 'community';
  const menu = document.getElementById('explorePaint');
  if (menu) menu.value = MAP_PAINT;
  const scale = document.getElementById('mapScale');
  if (scale) scale.hidden = MAP_PAINT !== 'ramp';
  if (!EXPLORER) return;
  EXPLORER.batch(() => {
    MAP_DATA.nodes.forEach((node, index) => {
      EXPLORER.getElementById('n' + index).data('fill', explorerFill(node));
    });
  });
}

// What one dot is filled with: its score on the library's ramp, or its community's
// colour. Either way nothing is the hollow dot, which is the panel a dot sits on.
function explorerFill(node) {
  const fill = MAP_PAINT === 'ramp' ? mapScoreFill(node) : mapCommunityColour(node.community);
  return explorerColour(fill || 'var(--panel)');
}

// ------------------------------------------------------------- the site's own numbers
//
// What a checkout cannot know about these games: what people scored them out of five and
// how much they have been played. `/api/graph` says in `overlays` whether there is a
// reading to fetch, so a page served by a machine with no store asks for nothing and no
// greyed-out control says what is missing. The overlay is keyed by the node hashes the
// base payload already drew, so merging it in needs no second coordinate system and no
// second opinion about which games exist.
//
// The overlay is one more source of size keys and the only source of the filters. It
// carries the shape the payload's own `sizes` carry, so the menu, the legend's sentence
// and a dot's radius go on working the way G11 built them.

const EXPLORER_OVERLAY = 'people';
const EXPLORER_OVERLAY_PATH = '/api/graph/people';

// Asked for once, and only where the payload says it can be answered. A failure is
// nothing rather than an error: the map draws, and the two keys and the three filters
// are simply not offered.
async function explorerPeopleOverlay() {
  const offers = (MAP_DATA && MAP_DATA.overlays) || [];
  if (offers.indexOf(EXPLORER_OVERLAY) < 0) return null;
  try {
    return await api(EXPLORER_OVERLAY_PATH);
  } catch (error) {
    return null;
  }
}

// What the overlay says about one game, by the node hash the map drew it under.
function explorerPeopleAt(hash) {
  const games = (EXPLORER_PEOPLE && EXPLORER_PEOPLE.games) || {};
  return games[hash] || null;
}

// ------------------------------------------------------------------------ the filters
//
// **A filter lights; it never hides**, which is G4's rule and is why these join the find
// box and the score menu rather than taking anything off the drawing. The names, the
// words and what each one reads are the server's: a filter is a key on a game's record
// and the page asks whether that key is true. One at a time, as the listing takes one.

function explorerFilters() {
  return (EXPLORER_PEOPLE && EXPLORER_PEOPLE.filters) || [];
}

function explorerKeepHtml() {
  const filters = explorerFilters();
  if (!filters.length) return '';
  const options = filters.map(one => `<option value="${esc(one.filter)}"${
    one.filter === EXPLORER_KEEP ? ' selected' : ''}>${esc(one.label)}</option>`).join('');
  return `<div class="row mapCtl">
    <label class="pick">and light up only
      <select id="exploreKeep" aria-label="which games to light up">
        <option value="">every game</option>${options}</select></label>
  </div>`;
}

// Whether one game answers the filter in hand. The key is the server's and the answer is
// a boolean on the game's own record, so the page names no filter and works out nothing
// about a game. A record the overlay has not got answers no filter, which is the same
// *absent is not zero* an unvoted game lives by.
function explorerKeeps(hash) {
  if (!EXPLORER_KEEP) return true;
  const one = explorerFilters().find(found => found.filter === EXPLORER_KEEP);
  const record = explorerPeopleAt(hash);
  return !!one && !!record && record[one.key] === true;
}

// The games the controls light, as one set, or null for *nothing is narrowing the map*.
// The map's own find box and score menu answer first and the filter narrows what they
// left, because two controls asking two questions about one library are answered by the
// games that satisfy both.
function explorerFound() {
  const found = mapFound();
  if (!EXPLORER_KEEP) return found;
  const kept = MAP_DATA.nodes
    .filter(node => explorerKeeps(node.hash) && (!found || found.has(node.hash)))
    .map(node => node.hash);
  return new Set(kept);
}

// ---------------------------------------------------------------------- size by data
//
// Which data draws the dots. The payload's `sizes` are the menu, in its own order, each
// with its rule in one sentence and its two real ends; `node.size` holds each dot's place
// between those ends as a number from 0 to 1. Nothing here decides what a size means:
// both ends and the sentence are the server's, exactly as the score ramp's two printed
// ends are. What is this page's is turning that number into a radius.

// The menu, in the servers' own order: the payload's keys first, then whatever the
// overlay adds. Two sources, one menu, because a reader is choosing what a dot says and
// not where the number came from.
function explorerSizes() {
  const own = (MAP_DATA && MAP_DATA.sizes) || [];
  return own.concat((EXPLORER_PEOPLE && EXPLORER_PEOPLE.sizes) || []);
}

// The key in effect, as the payload describes it.
function explorerSizeNow() {
  return explorerSizes().find(size => size.key === EXPLORER_SIZE) || null;
}

function explorerSizeHtml() {
  const sizes = explorerSizes();
  if (!sizes.length) return `<div class="row mapCtl mapEnds"><span></span>${explorerResetHtml()}</div>`;
  const options = sizes.map(size => `<option value="${esc(size.key)}"${
    size.key === EXPLORER_SIZE ? ' selected' : ''}>${esc(size.label)}</option>`).join('');
  return `<div class="row mapCtl">
    <label class="pick">size the dots by
      <select id="exploreSize" aria-label="what the size of a dot says">${
        options}</select></label>
    <span class="muted small mapSay" id="exploreSizeSay"></span>
    ${explorerResetHtml()}
  </div>`;
}

// What the current key means and what its two ends are. A size read against an imagined
// scale is read wrong, which is why the score ramp prints both of its ends, and a key
// some games have no value under has to say what happens to them.
function explorerSizeSay() {
  const said = document.getElementById('exploreSizeSay');
  const size = explorerSizeNow();
  if (!said) return;
  said.textContent = size
    ? `${size.words} The smallest dot is ${size.lowWords}, the largest ${size.highWords}.`
    : '';
}

// One dot's place between the current key's two ends. A dot with no value under the key
// -- a game with no current measurement, under *score* -- is drawn at the smallest size,
// which is what the sentence under the menu says it is.
// A key the overlay added is answered on the overlay's record for the same node, which
// is why it is keyed by the hash the map drew.
function explorerSizeValue(node) {
  const overlaid = explorerPeopleAt(node.hash);
  const held = (node.size || {});
  const at = EXPLORER_SIZE in held
    ? held[EXPLORER_SIZE]
    : (overlaid && overlaid.size || {})[EXPLORER_SIZE];
  return at === null || at === undefined ? 0 : at;
}

// How wide one dot is drawn. The range is G3's and the same under every key, so no key
// makes two dots touch: the layout promises one diameter between any two of them.
function explorerAcross(node) {
  const unit = (MAP_DATA.units && MAP_DATA.units.nodeRadius) || 1;
  const radius = EXPLORER_SMALLEST
    + (EXPLORER_LARGEST - EXPLORER_SMALLEST) * explorerSizeValue(node);
  return unit * radius * 2 * EXPLORER_UNIT;
}

// Switching the key resizes every dot and moves none. The ruled-out ring is the dot's own
// mark and takes its width with it, and the names are painted again because a name sits
// under the dot it belongs to. A folded group's dot keeps its own width: it stands for
// several games rather than one, and it carries a name that has to be read at the
// whole-map view.
function explorerPaintSizes() {
  if (!EXPLORER || !MAP_DATA) return;
  EXPLORER.batch(() => {
    MAP_DATA.nodes.forEach((node, index) => {
      const across = explorerAcross(node);
      EXPLORER.getElementById('n' + index).data('across', across);
      EXPLORER.getElementById('r' + index).data('across', across / 2);
    });
  });
  explorerPaintNames();
}

// The control over the picture as a whole: *reset* gives the server's picture back. It
// sits at the end of the dots' size row rather than on a row of its own. A community is
// folded into one dot from its own panel (`mapCommunityPanel`'s *fold it into one dot*),
// and opened again from the folded dot's: the communities are listed once, as the key,
// so the sheet does not push the drawing down with a second copy of them (Q801).
function explorerResetHtml() {
  return `<button id="exploreReset" title="Every community open, every game back where it
      started, and the lines the page opened with.">reset</button>`;
}

// Every game as a button, in the payload's own order, because a drawing on a canvas has
// no Tab stops of its own. Each says its name and its score in the words a card uses, and
// a game kept on purpose for what it shows says that after the score, as its card does.
// Seventy visible buttons would be a second map, so each is clipped where it sits and the
// focused one shows itself in the frame's corner.
function explorerKeysHtml() {
  const said = 'Every game on the map. Enter opens one, the arrow keys move the map,'
    + ' plus and minus zoom, 0 fits the whole library, L shows the game held up and its'
    + ' neighbours, a step further out each press, and Escape lets go.';
  return `<div class="exploreKeys" id="exploreKeys" role="group"
      aria-label="${esc(said)}">${MAP_DATA.nodes.map((node, index) => `
    <button data-game="${index}">${esc(node.name + ', ' + mapScoreWords(node)
      + (node.exhibit ? '. ' + node.exhibit : ''))}</button>`)
    .join('')}</div>`;
}

// The library, fetched once for the life of the page. Every later visit to the explorer
// finds `window.cytoscape` already there and draws without a request.
function explorerLibrary() {
  if (window.cytoscape) return Promise.resolve(true);
  if (EXPLORER_LOADING) return EXPLORER_LOADING;
  EXPLORER_LOADING = explorerScript(EXPLORER_SRC).then(loaded => loaded && !!window.cytoscape);
  return EXPLORER_LOADING;
}

// One vendored file, as a script tag: true once it ran, false when it did not arrive.
function explorerScript(file) {
  return new Promise(done => {
    const tag = document.createElement('script');
    tag.src = EXPLORER_VENDOR + file;
    tag.onload = () => done(true);
    tag.onerror = () => done(false);
    document.head.appendChild(tag);
  });
}

// The outlines' two files, fetched the first time the outlines are wanted and after the
// graph library, one after the other. Both register themselves with the library when they
// run, so what is asked afterwards is whether the drawing has learnt `bubbleSets`. A
// failure is forgotten, so the switch can ask again.
function explorerHullLibrary() {
  if (window.CytoscapeBubbleSets) return Promise.resolve(true);
  if (EXPLORER_HULL_LOADING) return EXPLORER_HULL_LOADING;
  EXPLORER_HULL_LOADING = explorerScript(EXPLORER_HULL_SRC[0])
    .then(first => first && explorerScript(EXPLORER_HULL_SRC[1]))
    .then(both => {
      const ready = !!both && !!window.CytoscapeBubbleSets;
      if (!ready) EXPLORER_HULL_LOADING = null;
      return ready;
    });
  return EXPLORER_HULL_LOADING;
}

// What a reader sees when the file does not arrive -- an extension that blocks scripts, a
// stale cache, a proxy. One sentence, and the way to the view that needs nothing.
function explorerSayBlocked() {
  EXPLORER_LOADING = null;
  const said = document.getElementById('exploreSay');
  if (said) said.textContent = '';
  const canvas = document.getElementById('explore');
  if (!canvas) return;
  canvas.innerHTML = `<div class="small u-p-18">The map needs a drawing
    library this site serves itself and your browser did not load it, so
    <a href="#/">the list</a> is the way to every game in it.</div>`;
}

// ------------------------------------------------------------------------- the colours
//
// Cytoscape draws on a canvas and knows nothing of custom properties, so every colour is
// resolved to what the theme currently makes of it. One probe element does it for all of
// them, which also means `color-mix` -- what the score ramp is written in -- arrives as
// the `rgb()` the library will take.

function explorerColour(value) {
  if (!EXPLORER_PROBE) {
    EXPLORER_PROBE = document.createElement('span');
    EXPLORER_PROBE.style.cssText = 'position:fixed; left:-9999px; top:0';
    document.body.appendChild(EXPLORER_PROBE);
  }
  EXPLORER_PROBE.style.color = '';
  EXPLORER_PROBE.style.color = value;
  return explorerPixel(getComputedStyle(EXPLORER_PROBE).color);
}

// The same colour as the three numbers the library will take. A browser answers a
// `color-mix` with `oklab(...)`, which the score ramp is written in and Cytoscape's own
// colour reader has never heard of, so the answer is read off a pixel painted in it
// rather than parsed. One pixel, painted once per dot as the elements are built.
function explorerPixel(colour) {
  if (!EXPLORER_INK) {
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    EXPLORER_INK = canvas.getContext('2d', { willReadFrequently: true });
  }
  EXPLORER_INK.clearRect(0, 0, 1, 1);
  EXPLORER_INK.fillStyle = colour;
  EXPLORER_INK.fillRect(0, 0, 1, 1);
  const seen = EXPLORER_INK.getImageData(0, 0, 1, 1).data;
  return `rgb(${seen[0]}, ${seen[1]}, ${seen[2]})`;
}

// ------------------------------------------------------------------------ the elements
//
// One entry per dot and one per line, with everything the stylesheet reads carried as
// data. Built once from the payload; nothing here is recomputed on a pan, a zoom or a tap.

function explorerPlace(node) {
  return { x: node.x * EXPLORER_UNIT, y: node.y * EXPLORER_UNIT };
}

function explorerElements() {
  const parts = [];
  MAP_DATA.nodes.forEach((node, index) => {
    const across = explorerAcross(node);
    parts.push({
      group: 'nodes',
      data: {
        id: 'n' + index, at: index, kind: 'node', across: across, name: node.name,
        // A game with no score this checkout can read, or in no coloured community, is
        // the hollow dot the map draws: the panel's own colour, which is the background
        // a dot sits on.
        fill: explorerFill(node),
        // An unusual game is in no community, and its edge says so.
        edge: explorerColour(node.community ? 'var(--ink)' : 'var(--muted)'),
      },
      position: explorerPlace(node),
    });
    // The ruled-out ring, inside the dot's own radius. A second element because one node
    // carries one border, and the layout promises only one diameter between two dots, so
    // a ring drawn outside two neighbours would meet in the middle.
    if (node.score && node.score.viable === false) {
      parts.push({
        group: 'nodes',
        data: {
          id: 'r' + index, at: index, kind: 'ring', across: across / 2,
          fill: explorerColour('var(--panel)'), edge: explorerColour('var(--bad)'),
        },
        position: explorerPlace(node),
        classes: 'ring',
        selectable: false,
        // It is the dot's mark rather than a thing of its own, so it is not dragged; it
        // follows the dot it marks, wherever a drag or a *reset* takes that.
        grabbable: false,
      });
    }
  });
  // One dot per community that folds, drawn where its games average out, in its colour,
  // and hidden until it is folded. It is the same element either way, so folding and
  // opening move nothing and add nothing.
  mapGroups().forEach((group, index) => {
    parts.push({
      group: 'nodes',
      data: {
        id: 'g' + index, at: index, kind: 'group', name: group.name,
        across: EXPLORER_GROUP,
        fill: explorerColour(mapCommunityColour(group.id) || 'var(--panel)'),
        edge: explorerColour('var(--ink)'),
      },
      position: explorerPlace(group),
      classes: 'group away',
    });
  });
  // The lines, every row the payload sent, as wide as the two games are alike. Added
  // first so an arrow, when the arrows are on, lies on top.
  mapLines().forEach((line, index) => {
    const from = MAP_AT[line.source], to = MAP_AT[line.target];
    if (!from || !to) return;
    parts.push({
      group: 'edges',
      data: {
        id: 's' + index, at: index, kind: 'line', likeness: line.likeness,
        source: 'n' + MAP_DATA.nodes.indexOf(from),
        target: 'n' + MAP_DATA.nodes.indexOf(to),
        colour: explorerColour(mapLineColour(line)),
      },
      classes: 'line',
    });
  });
  MAP_DATA.lineage.forEach((arrow, index) => {
    const from = MAP_AT[arrow.source], to = MAP_AT[arrow.target];
    if (!from || !to) return;
    parts.push({
      group: 'edges',
      data: {
        id: 'a' + index, at: index, kind: 'arrow',
        source: 'n' + MAP_DATA.nodes.indexOf(from),
        target: 'n' + MAP_DATA.nodes.indexOf(to),
        colour: explorerColour('var(--muted)'),
      },
      classes: 'lineage',
    });
  });
  return parts;
}

// The two kinds of line, the fill, the ring and the names, as the library's own
// stylesheet. A line of likeness is as wide as the two games are alike, in their
// community's colour, and grey with an arrowhead at the middle is *made from*. The
// arrowhead sits at the middle rather than the end because at the end it would be under
// the dot it points at.
function explorerStyle() {
  const mark = explorerColour('var(--hl-sel)');
  const faint = explorerColour('var(--faint)');
  const ink = explorerColour('var(--ink)');
  const back = explorerColour('var(--bg)');
  return [
    { selector: 'node', style: {
      'width': 'data(across)', 'height': 'data(across)',
      'background-color': 'data(fill)',
      'border-width': 1.4, 'border-color': 'data(edge)',
      'label': '',
    } },
    // A name, on the dots `explorerPaintNames` found room for. The size is in model units
    // and is rewritten as the zoom changes, so a name is the same size on screen wherever
    // the reader is -- which is the map's own rule, and what makes the question *does
    // this name fit* one a pass over the drawing can answer.
    { selector: 'node.named', style: {
      'label': 'data(name)', 'font-size': 'data(font)',
      'text-valign': 'bottom', 'text-margin-y': 2, 'text-wrap': 'none',
      'color': ink, 'text-outline-color': back, 'text-outline-width': 'data(halo)',
      'text-events': 'no',
    } },
    // A folded group. A rounded box rather than a circle, because it is not a game and
    // the difference has to be visible at the whole-map view, where its name may be the
    // only thing a reader can read.
    { selector: 'node.group', style: {
      'shape': 'round-rectangle', 'border-width': 2.4,
    } },
    { selector: 'node.ring', style: {
      'background-opacity': 0,
      'border-width': 1.4, 'border-color': explorerColour('var(--bad)'),
      // It marks the dot underneath it and answers nothing itself, so a tap falls
      // through to the game, and the name belongs to the game rather than to the mark.
      'events': 'no', 'label': '',
    } },
    { selector: 'edge', style: {
      'width': 1.7, 'line-color': 'data(colour)', 'curve-style': 'straight',
      'line-cap': 'round',
    } },
    // A line: as wide as its two games are alike, in their community's colour or the
    // neutral one. The width is the server's number, mapped; nothing here works one out.
    { selector: 'edge.line', style: { 'width': EXPLORER_WIDTH } },
    // A folded group's line to the rest of the library: the lines its games had, drawn
    // once and labelled with how many they were, in grey.
    { selector: 'edge.grouped', style: {
      'width': 2.6, 'label': 'data(count)', 'font-size': 'data(font)',
      'color': ink, 'text-outline-color': back, 'text-outline-width': 'data(halo)',
      'text-events': 'no',
    } },
    { selector: 'edge.lineage', style: {
      'width': 1.5,
      'mid-target-arrow-shape': 'triangle',
      'mid-target-arrow-color': 'data(colour)',
      'arrow-scale': 0.9,
    } },
    // What is let go of while something else is held up, or while a filter lights part of
    // the library, or while a pointer rests on a game (`hushed`). A colour rather than an
    // opacity, which is the map's rule: a faded hairline is still exactly a hairline.
    { selector: '.faded, .hushed', style: {
      'border-color': faint, 'line-color': faint, 'color': faint,
      'mid-target-arrow-color': faint,
    } },
    // A game the find box or the score menu lights.
    { selector: 'node.found', style: { 'border-width': 2.2, 'border-color': mark } },
    // The one a pick holds up, the one the pointer rests on, and the find box's first
    // match.
    { selector: 'node.picked, node.hovered, node.held', style: {
      'border-width': 2.6, 'border-color': mark,
    } },
    { selector: 'edge.picked', style: { 'width': 3.4, 'line-color': mark } },
    { selector: 'edge.picked.line', style: { 'width': EXPLORER_WIDTH } },
    { selector: 'edge.picked.lineage', style: { 'mid-target-arrow-color': mark } },
    // The four things that take something off the drawing: the arrows' switch (`off`),
    // *hide the rest*, a fold -- which takes away the games it folded, the group dots of
    // every group still open, and the lines a folded dot now carries instead -- and the
    // local view (`aside`). A line with an end that is not displayed is not drawn, which
    // is the library's own rule and why hiding a game needs nothing said about its lines.
    { selector: '.off, .hidden, .away, .aside', style: { 'display': 'none' } },
  ];
}

// ------------------------------------------------------------------------- the drawing

function explorerDraw() {
  const canvas = document.getElementById('explore');
  if (!canvas || !MAP_DATA) return;
  EXPLORER = cytoscape({
    container: canvas,
    elements: explorerElements(),
    style: explorerStyle(),
    // The server's picture, taken as given. Every visitor opens on the same one.
    layout: { name: 'preset', fit: true, padding: EXPLORER_AIR },
    // Wheel, drag and pinch are the library's own, and so is dragging one dot now that
    // *reset* can undo it.
    boxSelection: false,
    selectionType: 'single',
    // Low enough that the whole map, and a local view under its bar, fit a phone's frame.
    minZoom: 0.05,
    maxZoom: 9,
  });
  explorerFitWhenMeasured();
  explorerSay();
  EXPLORER.on('zoom pan resize', explorerNamesSoon);
  explorerRingsFollow();
  EXPLORER.on('tap', event => {
    const target = event.target;
    if (target === EXPLORER) return explorerPick(null);
    const held = target.data();
    if (held.kind === 'node') return explorerPick({ kind: 'node', at: held.at });
    if (held.kind === 'line') return explorerPick({ kind: 'line', at: held.at });
    if (held.kind === 'arrow') return explorerPick({ kind: 'arrow', at: held.at });
    // A folded dot, and the lines it carries, both answer for the group.
    if (held.kind === 'group') return explorerPick({ kind: 'group', at: held.at });
    if (held.kind === 'grouped') return explorerPick({ kind: 'group', at: held.group });
  });
  // A pointer resting on a game lights it and its neighbours at once; the tap is still
  // what holds them up and opens the panel. A touch screen has no resting pointer, and
  // there the tap does both.
  EXPLORER.on('mouseover', 'node', event => explorerHover(event.target));
  EXPLORER.on('mouseout', 'node', () => explorerHushOff());
  document.getElementById('exploreIn').onclick = () => explorerZoom(EXPLORER_STEP);
  document.getElementById('exploreOut').onclick = () => explorerZoom(1 / EXPLORER_STEP);
  document.getElementById('exploreHome').onclick = () => {
    explorerPick(null);
    EXPLORER.fit(undefined, EXPLORER_AIR);
  };
}

// The ruled-out ring is a node of its own at the dot's place, because one node carries
// one border and a ruled-out game needs a second mark inside its own radius. So whatever
// moves a dot has to move its ring, and this is the one place that knows it: a drag and a
// *reset* both end in a position, and a position is what this watches.
function explorerRingsFollow() {
  EXPLORER.on('position', 'node', event => {
    const dot = event.target;
    if (dot.data('kind') !== 'node') return;
    const ring = EXPLORER.getElementById('r' + dot.data('at'));
    if (ring.length) ring.position({ x: dot.position('x'), y: dot.position('y') });
  });
}

// The frame is written into the page in the same turn as the drawing, so the browser may
// not have laid it out by the time the library measures the container -- and a picture
// fitted to a container of no size opens at zoom 1 with the whole library off screen.
// Fitted the first time the frame has a size, and once. A window resize after that is the
// library's own; it watches for one.
function explorerFitWhenMeasured() {
  const canvas = EXPLORER.container();
  const fit = () => { EXPLORER.resize(); EXPLORER.fit(undefined, EXPLORER_AIR); };
  const measured = () => {
    const box = canvas.getBoundingClientRect();
    return !!(box.width && box.height);
  };
  if (measured()) return fit();
  const watching = new ResizeObserver(() => {
    if (!measured()) return;
    watching.disconnect();
    fit();
  });
  watching.observe(canvas);
}

// What is drawn, counted out of the payload rather than off the canvas, so the sentence
// says what the server sent and a line the drawing lost would show up as a difference.
function explorerSay() {
  const said = document.getElementById('exploreSay');
  if (!said) return;
  const counts = MAP_DATA.counts;
  said.textContent = `${counts.nodes} games in ${counts.communities} communities, and`
    + ` ${counts.unusual} unusual ones in none. The thicker a line, the more alike its two`
    + ' games. Point at a game to light its neighbours, or tap one for the games like it.';
}

function explorerZoom(factor) {
  if (!EXPLORER) return;
  const box = EXPLORER.container().getBoundingClientRect();
  EXPLORER.zoom({
    level: EXPLORER.zoom() * factor,
    renderedPosition: { x: box.width / 2, y: box.height / 2 },
  });
}

// --------------------------------------------------------------------------- the names
//
// Which names are painted, decided fresh whenever the view moves. G3 found the name to be
// the scarce thing on a map of seventy games: at the whole-map view about half of them can
// carry one, so the order is the map's own -- whatever is held up first, then the busiest
// games -- a name whose box would land on one already painted is left off, and zooming in
// makes room for the ones that were.
//
// Nothing here moves a dot. What it writes is a class and the size that class draws at.

function explorerPaintNames() {
  if (!EXPLORER || !MAP_DATA) return;
  const frame = EXPLORER.container().getBoundingClientRect();
  const at = EXPLORER.zoom();
  if (!frame.width || !at) return;
  const taken = explorerHullNames(frame);
  const named = [];
  const room = part => {
    const place = part.renderedPosition();
    const wide = String(part.data('name')).length * EXPLORER_TEXT_PX * EXPLORER_TEXT_WIDTH
      + EXPLORER_TEXT_PX * EXPLORER_TEXT_AIR;
    return {
      x: place.x - wide / 2, y: place.y + part.renderedHeight() / 2,
      w: wide, h: EXPLORER_TEXT_PX * 1.25,
    };
  };
  // A folded dot is named first and always: it stands for several games, and one without
  // its name is a dot nobody can read.
  EXPLORER.nodes('.group').forEach(part => {
    if (part.hasClass('away') || part.hasClass('hidden') || part.hasClass('aside')) return;
    taken.push(room(part));
    named.push(part);
  });
  // What the reader is pointing at, or typed, is named before anything else.
  const first = new Set(EXPLORER_NEAR || []);
  if (EXPLORER_FIRST) first.add(EXPLORER_FIRST);
  mapLabelOrder(first).forEach(index => {
    const part = EXPLORER.getElementById('n' + index);
    const node = MAP_DATA.nodes[index];
    if (!part.length || !node) return;
    if (part.hasClass('hidden') || part.hasClass('away') || part.hasClass('aside')) return;
    const box = room(part);
    // Whole names only: half a name against the frame's edge reads as a different word.
    if (box.x < 0 || box.x + box.w > frame.width) return;
    if (box.y < 0 || box.y + box.h > frame.height) return;
    if (taken.some(other => mapOverlaps(other, box))) return;
    taken.push(box);
    named.push(part);
  });
  EXPLORER.batch(() => {
    EXPLORER.nodes('.named').removeClass('named');
    named.forEach(part => {
      // Screen pixels into the model units the library paints a label in.
      part.data('font', EXPLORER_TEXT_PX / at);
      part.data('halo', EXPLORER_TEXT_PX * 0.2 / at);
      part.addClass('named');
    });
    // A folded dot's lines carry a count, which is a label and wants the same treatment.
    EXPLORER.edges('.grouped')
      .data('font', EXPLORER_TEXT_PX / at)
      .data('halo', EXPLORER_TEXT_PX * 0.2 / at);
  });
}

// A view change is a pan, a zoom, a pinch or a window resize, and each of them arrives
// many times a second. One pass per frame is enough, and the pass reads the view rather
// than the event.
function explorerNamesSoon() {
  if (explorerNamesSoon.waiting) return;
  explorerNamesSoon.waiting = true;
  requestAnimationFrame(() => {
    explorerNamesSoon.waiting = false;
    explorerPaintNames();
  });
}

// -------------------------------------------------------------- communities that fold
//
// A community folds into one dot and opens again. Its games are taken off the drawing
// where they stand, its own dot is put on, and every line one of its games had to a game
// outside is drawn once, from the folded dot, labelled with how many lines it stands for.
// Two folded communities share one line between them. Nothing else moves, and opening
// the community puts every game back where it was, because it was never moved.
//
// The payload's `communities` are a partition -- a game folds into one place or none --
// and `mapGroups` is the ones of three or more, which are the ones a fold chip offers.

// Where each game is drawn: its own dot, or the folded dot of the group holding it.
function explorerHomes() {
  const home = {};
  MAP_DATA.nodes.forEach((node, index) => { home[node.hash] = 'n' + index; });
  mapGroups().forEach((group, index) => {
    if (!EXPLORER_FOLDED.has(group.id)) return;
    group.members.forEach(hash => { home[hash] = 'g' + index; });
  });
  return home;
}

function explorerFoldPaint() {
  if (!EXPLORER || !MAP_DATA) return;
  const home = explorerHomes();
  EXPLORER.batch(() => {
    EXPLORER.remove(EXPLORER.edges('.grouped'));
    MAP_DATA.nodes.forEach((node, index) => {
      const away = home[node.hash] !== 'n' + index;
      EXPLORER.getElementById('n' + index).toggleClass('away', away);
      EXPLORER.getElementById('r' + index).toggleClass('away', away);
    });
    mapGroups().forEach((group, index) => {
      EXPLORER.getElementById('g' + index)
        .toggleClass('away', !EXPLORER_FOLDED.has(group.id));
    });
    EXPLORER.add(explorerGroupLines(home));
  });
}

// The lines a folded dot carries. One per pair of ends, counting the lines it stands for;
// a line between two games of one folded group is gone, which is what folding them into
// one dot says about it. An arrow is counted only while the arrows are drawn, so the
// switch and a fold read together rather than against each other.
function explorerGroupLines(home) {
  const made = new Map();
  const gather = (line, index, kind) => {
    const first = home[line.source], second = home[line.target];
    if (!first || !second) return;
    const folded = first[0] === 'g' || second[0] === 'g';
    EXPLORER.getElementById({ arrow: 'a', line: 's' }[kind] + index)
      .toggleClass('away', folded);
    if (!folded || first === second) return;
    // An arrow keeps its direction; a line has none, so its two ends are sorted and the
    // pair is one line whichever way round it was found.
    const ends = kind === 'arrow' || first < second
      ? [first, second] : [second, first];
    const key = kind + '|' + ends[0] + '|' + ends[1];
    const held = made.get(key) || { kind: kind, source: ends[0], target: ends[1], count: 0 };
    held.count += 1;
    made.set(key, held);
  };
  MAP_DATA.lineage.forEach((arrow, index) => {
    if (mapArrowHidden()) {
      EXPLORER.getElementById('a' + index).toggleClass('away', false);
      return;
    }
    gather(arrow, index, 'arrow');
  });
  mapLines().forEach((line, index) => gather(line, index, 'line'));

  const drawn = [];
  made.forEach(held => {
    const at = held.source[0] === 'g' ? held.source : held.target;
    drawn.push({
      group: 'edges',
      data: {
        id: 'x' + drawn.length, kind: 'grouped', group: Number(at.slice(1)),
        source: held.source, target: held.target, count: held.count,
        colour: explorerColour('var(--muted)'),
      },
      classes: held.kind === 'arrow' ? 'grouped lineage' : 'grouped',
    });
  });
  return drawn;
}

// Folding one group, or opening it. A pick that the press takes off the drawing moves to
// what replaced it: a game in a group that has just folded, or the community lit when its
// panel's *fold it into one dot* is pressed, is held up as that group, and a group that
// has just opened lets go.
function explorerFoldToggle(index) {
  const group = mapGroupAt(index);
  if (!group) return;
  const folding = !EXPLORER_FOLDED.has(group.id);
  if (folding) EXPLORER_FOLDED.add(group.id); else EXPLORER_FOLDED.delete(group.id);
  const held = MAP_PICK && MAP_PICK.kind === 'node' && MAP_DATA.nodes[MAP_PICK.at];
  const lit = MAP_PICK && MAP_PICK.kind === 'community'
    && (mapCommunityAt(MAP_PICK.at) || {}).id === group.id;
  if (folding && (lit || (held && group.members.includes(held.hash)))) {
    MAP_PICK = { kind: 'group', at: index };
    MAP_ON = mapLit();
  } else if (!folding && MAP_PICK && MAP_PICK.kind === 'group' && MAP_PICK.at === index) {
    MAP_PICK = null;
    MAP_ON = null;
  }
  explorerFoldDone();
}

// Everything a fold changes, in one pass: the chips, what is drawn, what is held up and
// the panel. The lines are repainted first because which of them are hidden decides which
// the folded dot counts.
function explorerFoldDone() {
  explorerPaintCommunityChips();
  explorerPaintLines();
  explorerFoldPaint();
  explorerPaintPick();
  explorerPaintLit();
  explorerPaintPanel();
}

// A game somewhere in a folded group, wanted by name: the group opens first, because a
// dot nobody can see cannot be held up or centred.
function explorerUnfold(hash) {
  const at = mapGroups().findIndex(group =>
    EXPLORER_FOLDED.has(group.id) && group.members.includes(hash));
  if (at >= 0) explorerFoldToggle(at);
}

// ------------------------------------------------------------------------ the outlines
//
// `docs/archive/plan-graph-map.md` S5. A smooth outline round each community of three or
// more, in its colour, with the server's name for it above. The outline is drawn by the
// vendored bubble sets on an SVG layer under the dots, from where the drawn members and
// the likeness lines between them are, keeping clear of the drawn games that are not
// members; it follows a drag on its own. Which games are in a community is the server's,
// and nothing here decides it: what is decided is which of them are on screen.
//
// An outline is drawn again only when what it would hold has changed -- its members on
// screen and the lines drawn between them -- so a pick or a hover, which change neither,
// cost nothing but a colour.

// The switch. The first time it is on, the two files are fetched; a failure turns it off
// and says so on the switch itself, and the map goes on without outlines.
async function explorerSetHulls(on) {
  EXPLORER_HULLS = !!on;
  const box = document.getElementById('exploreHulls');
  if (box) box.checked = EXPLORER_HULLS;
  if (EXPLORER_HULLS && !(await explorerHullLibrary())) {
    EXPLORER_HULLS = false;
    if (box) {
      box.checked = false;
      box.parentElement.title = 'The outlines need two small files this site serves'
        + ' itself, and your browser did not load them.';
    }
  }
  explorerPaintHulls();
}

// The outlines one community is drawn with: one for each place the server says it is
// drawn (`parts`), the place holding its centre first, each under an id of its own. A
// community the payload gives no parts is one place.
function explorerHullParts(community) {
  const parts = community.parts && community.parts.length
    ? community.parts : [community.members];
  return parts.map((members, at) => ({ id: community.id + '#' + at, members }));
}

// What each outline holds now: the games of one place of a community that are drawn --
// not folded, not set aside by the local view, not hidden -- and the drawn lines between
// them. A folded community is its dot and has no outline; a place with fewer than
// `EXPLORER_HULL_LEAST` games on screen has none either.
function explorerHullsWanted() {
  const wanted = new Map();
  if (!EXPLORER_HULLS) return wanted;
  const shown = EXPLORER.nodes('[kind = "node"]').not('.away, .aside, .hidden');
  mapGroups().forEach(community => {
    if (EXPLORER_FOLDED.has(community.id)) return;
    explorerHullParts(community).forEach(place => {
      const inside = new Set(place.members);
      const members = shown.filter(part => inside.has(MAP_DATA.nodes[part.data('at')].hash));
      if (members.length < EXPLORER_HULL_LEAST) return;
      const lines = members.edgesWith(members).filter('.line').not('.off, .away');
      const ids = parts => parts.map(part => part.id()).sort().join(' ');
      wanted.set(place.id, {
        community, members, lines, others: shown.not(members),
        key: ids(members) + '|' + ids(lines),
      });
    });
  });
  return wanted;
}

// Every outline brought up to date with what is drawn: one whose holding has not changed
// is left alone, one whose has is drawn again, and one no longer wanted goes.
function explorerPaintHulls() {
  if (!EXPLORER) return;
  if (EXPLORER_HULLS && !EXPLORER_BUBBLES && typeof EXPLORER.bubbleSets === 'function') {
    EXPLORER_BUBBLES = EXPLORER.bubbleSets(EXPLORER_HULL_REACH);
    explorerWatchHulls();
  }
  const wanted = EXPLORER_BUBBLES ? explorerHullsWanted() : new Map();
  EXPLORER_OUTLINES.forEach((held, id) => {
    const want = wanted.get(id);
    if (want && want.key === held.key) return;
    EXPLORER_BUBBLES.removePath(held.path);
    held.label.remove();
    EXPLORER_OUTLINES.delete(id);
  });
  wanted.forEach((want, id) => {
    if (EXPLORER_OUTLINES.has(id)) return;
    const colour = mapCommunityColour(want.community.id);
    const path = EXPLORER_BUBBLES.addPath(want.members, want.lines, want.others, {
      className: 'mapHull',
      style: {
        fill: colour, fillOpacity: '0.12', stroke: colour, strokeWidth: '1.6',
        // The stroke keeps its width at every zoom, as a dot's border does.
        vectorEffect: 'non-scaling-stroke',
      },
    });
    EXPLORER_OUTLINES.set(id, {
      path, key: want.key, members: want.members, colour,
      label: explorerHullLabel(want.community, colour),
    });
  });
  // The names go on top of every outline, so the last one drawn does not cover them.
  EXPLORER_OUTLINES.forEach(held => EXPLORER_BUBBLES.layer.node.appendChild(held.label));
  explorerHullsFade();
  explorerNamesSoon();
}

// A drag moves an outline on its own, a moment later; its name follows when the outline
// has been redrawn.
function explorerWatchHulls() {
  new MutationObserver(() => explorerNamesSoon()).observe(EXPLORER_BUBBLES.layer.node, {
    attributes: true, attributeFilter: ['d'], subtree: true,
  });
}

// A community's name as SVG text on the outlines' own layer, broken into lines of about
// `EXPLORER_HULL_LINE` characters at the spaces, led by a dot in the outline's colour: on a
// crowded view a name can sit nearer another community's outline than its own, and the
// dot is what says whose it is. Where it goes is `explorerHullPlace`'s.
function explorerHullLabel(community, colour) {
  const svg = 'http://www.w3.org/2000/svg';
  const text = document.createElementNS(svg, 'text');
  text.setAttribute('class', 'mapHullName');
  text.setAttribute('text-anchor', 'middle');
  const lines = [];
  String(community.name).split(' ').forEach(word => {
    const last = lines.length - 1;
    if (last >= 0 && (lines[last] + ' ' + word).length <= EXPLORER_HULL_LINE) {
      lines[last] += ' ' + word;
    } else {
      lines.push(word);
    }
  });
  lines.forEach((line, at) => {
    const span = document.createElementNS(svg, 'tspan');
    span.setAttribute('dy', at ? '1.15em' : '0');
    if (!at) {
      const dot = document.createElementNS(svg, 'tspan');
      dot.setAttribute('class', 'mapHullDot');
      dot.style.fill = colour;
      dot.textContent = '● ';
      span.appendChild(dot);
    }
    span.appendChild(document.createTextNode(line));
    text.appendChild(span);
  });
  EXPLORER_BUBBLES.layer.node.appendChild(text);
  return text;
}

// One name centred on its outline, at the map's one text size whatever the zoom -- the
// layer is scaled with the drawing, so the size is set again in model units on every view
// change, as a dot's name is. Above the outline, its last line just clear of the top, or
// below it, its first line just clear of the bottom. False when the outline has no size.
function explorerHullPlace(held, below) {
  const size = EXPLORER_TEXT_PX / EXPLORER.zoom();
  let box = null;
  try { box = held.path.node.getBBox(); } catch (error) { box = null; }
  if (!box || !box.width) return false;
  const x = box.x + box.width / 2;
  const lines = held.label.childNodes.length;
  held.label.setAttribute('font-size', String(size));
  held.label.setAttribute('stroke-width', String(size * 0.3));
  held.label.setAttribute('y', String(below
    ? box.y + box.height + size * 1.1
    : box.y - size * (0.45 + 1.15 * (lines - 1))));
  held.label.childNodes.forEach(span => span.setAttribute('x', String(x)));
  return true;
}

// The names the outlines carry, kept where they fit: whole, inside the frame, and clear of
// a name already kept, biggest community first and each community's own places in the
// server's order -- above the outline, else below it, else not at all until a zoom makes
// room. What is kept is handed back as the first boxes the games' own names must stay
// clear of.
function explorerHullNames(frame) {
  const taken = [];
  if (!EXPLORER_OUTLINES.size) return taken;
  const origin = EXPLORER.container().getBoundingClientRect();
  const room = element => {
    const seen = element.getBoundingClientRect();
    return { x: seen.left - origin.left, y: seen.top - origin.top,
             w: seen.width, h: seen.height };
  };
  // The local view's bar lies over the drawing's top edge, and a name under it is not read.
  const bar = document.getElementById('exploreLocal');
  if (bar && !bar.hidden) taken.push(room(bar));
  // A name is drawn under the dots, so one laid over a game would be half hidden by it and
  // would hide where the game is: the drawn games are boxes a name keeps clear of too.
  const games = EXPLORER.nodes('[kind = "node"]').not('.away, .aside, .hidden').map(part => {
    const seen = part.renderedBoundingBox({ includeLabels: false });
    return { x: seen.x1, y: seen.y1, w: seen.w, h: seen.h };
  });
  const fits = box => box.w > 0 && box.x >= 0 && box.y >= 0
    && box.x + box.w <= frame.width && box.y + box.h <= frame.height
    && !taken.some(other => mapOverlaps(other, box))
    && !games.some(other => mapOverlaps(other, box));
  mapGroups().forEach(community => explorerHullParts(community).forEach(place => {
    const held = EXPLORER_OUTLINES.get(place.id);
    if (!held) return;
    held.label.removeAttribute('display');
    const kept = [false, true].some(below => {
      if (!explorerHullPlace(held, below)) return false;
      const box = room(held.label);
      if (!fits(box)) return false;
      taken.push(box);
      return true;
    });
    if (!kept) held.label.setAttribute('display', 'none');
  }));
  return taken;
}

// An outline is as lit as its games: while a pick, a filter or a resting pointer fades
// every one of them, the outline and its name fade too, to the map's one faded colour.
function explorerHullsFade() {
  EXPLORER_OUTLINES.forEach(held => {
    const lit = held.members.some(part => !part.hasClass('faded') && !part.hasClass('hushed'));
    const colour = lit ? held.colour : 'var(--faint)';
    held.path.node.style.stroke = colour;
    held.path.node.style.fill = colour;
    held.label.classList.toggle('faded', !lit);
  });
}

// ------------------------------------------------------------------------------ reset
//
// The control over the picture as a whole: the server's coordinates, every group open,
// and the lines the page opened with.
//
// There was a *tidy* beside it, the graph library's own force layout over what was on
// screen, and Q477 retired it. The library's layout pulls every dot towards one point of
// the frame and packs whatever is not joined into rows, so over the whole map it crossed
// ten pairs of outlines; and the server's picture is the one that keeps games alike near
// each other and stays where a returning reader left it (`docs/plan-map.md` M3). A layout
// here would be another picture, drawn by other rules and with none of its promises. A
// reader who has dragged the picture out of shape has *reset*.

function explorerReset() {
  if (!EXPLORER) return;
  EXPLORER_FOLDED = new Set();
  EXPLORER_LOCAL = null;
  // The switch the map opened with.
  MAP_ARROWS = false;
  const box = document.getElementById('exploreArrows');
  if (box) box.checked = MAP_ARROWS;
  const key = document.getElementById('mapArrowsKey');
  if (key) key.hidden = !MAP_ARROWS;
  explorerSetPaint('community');
  explorerSetHulls(EXPLORER_HULLS_OPEN);
  EXPLORER.batch(() => {
    MAP_DATA.nodes.forEach((node, index) => {
      EXPLORER.getElementById('n' + index).position(explorerPlace(node));
    });
    mapGroups().forEach((group, index) => {
      EXPLORER.getElementById('g' + index).position(explorerPlace(group));
    });
  });
  MAP_PICK = null;
  MAP_ON = null;
  explorerFoldDone();
  explorerHullsAfresh();
  EXPLORER.fit(undefined, EXPLORER_AIR);
}

// Every outline worked out again from nothing, over the picture as it is now. An outline
// redraws itself when one of its own games moves, but from what it kept of the games it
// kept clear of, so after a dot is dragged across another community and put back, that
// community's outline stays dented -- in the Browser pane, round the space the dot left,
// with one of its own games outside it. Only a forced redraw forgets what it kept, and it
// has to come after the library's own redraws of the move, which wait up to
// `EXPLORER_HULL_SETTLE`; one before them is undone by them.
function explorerHullsAfresh() {
  setTimeout(() => {
    if (EXPLORER_BUBBLES) EXPLORER_BUBBLES.update(true);
  }, EXPLORER_HULL_SETTLE);
}

// --------------------------------------------------------------- hiding the arrows
//
// The arrows' switch reaches the arrows and nothing else. Every position arrived with the
// payload and was written once, and the paint the switch calls selects no dot at all.

function explorerPaintLines() {
  if (!EXPLORER) return;
  EXPLORER.batch(() => {
    EXPLORER.edges().forEach(part => {
      const held = part.data();
      // A folded dot's own line is built from the lines that are showing, so it is never
      // one of the hidden ones; and every line of likeness is drawn.
      if (held.kind !== 'arrow') return;
      part.toggleClass('off', mapArrowHidden());
    });
  });
}

// ------------------------------------------------------- what is lit, and what is faded
//
// One pass over the payload, run whenever the pick or the filter changes. A pick lights
// one game and its lines; the find box and the score menu light a set of games by what
// the payload says about them. Everything else fades to `--faint`, which is a colour and
// never an opacity.
//
// **A filter lights; it never hides** -- unless the reader asked for *hide the rest*, and
// then the unlit games go and their lines go with them, because the library does not draw
// a line whose end is not displayed.

function explorerPaintLit() {
  if (!EXPLORER) return;
  const found = explorerFound();
  const lit = hash => !found || found.has(hash);
  EXPLORER.batch(() => {
    MAP_DATA.nodes.forEach((node, index) => {
      const shown = lit(node.hash);
      const up = !MAP_ON || MAP_ON.nodes.has(node.hash);
      const dot = EXPLORER.getElementById('n' + index);
      const both = dot.union(EXPLORER.getElementById('r' + index));
      both.toggleClass('faded', !up || (!MAP_PICK && !shown));
      both.toggleClass('hidden', EXPLORER_ONLY && !shown);
      dot.toggleClass('found', !MAP_PICK && !!found && shown);
    });
    // A line with both ends lit keeps its colour, which is the reading somebody who
    // narrowed the map to four games is asking for. Every other line fades, so the shape
    // is still there and the lit games are what the eye lands on.
    mapLines().forEach((line, index) => {
      const ends = lit(line.source) && lit(line.target);
      const part = EXPLORER.getElementById('s' + index);
      part.toggleClass('faded', MAP_ON ? !MAP_ON.lines.has(index) : !ends);
    });
    MAP_DATA.lineage.forEach((arrow, index) => {
      const ends = lit(arrow.source) && lit(arrow.target);
      const part = EXPLORER.getElementById('a' + index);
      part.toggleClass('faded', MAP_ON ? !MAP_ON.arrows.has(index) : !ends);
    });
    // A folded dot stands for its games, so it is lit when any of them is and it goes
    // under *hide the rest* only when none of them is left.
    mapGroups().forEach((group, index) => {
      const shown = group.members.some(hash => lit(hash));
      const up = !MAP_ON || group.members.some(hash => MAP_ON.nodes.has(hash));
      const dot = EXPLORER.getElementById('g' + index);
      dot.toggleClass('faded', !up || (!MAP_PICK && !shown));
      dot.toggleClass('hidden', EXPLORER_ONLY && !shown);
      dot.toggleClass('found', !MAP_PICK && !!found && shown);
    });
    // A line a folded dot carries follows its two ends: it is as lit as the less lit of
    // them, which is the same rule as a line between two games.
    EXPLORER.edges('.grouped').forEach(part => {
      part.toggleClass('faded',
        part.source().hasClass('faded') || part.target().hasClass('faded'));
    });
  });
  // The local view follows whatever lines are drawn now, so it is worked out again last.
  explorerPaintLocal();
  // Whatever is lit is named first, so the names follow the pick and the filter.
  explorerPaintNames();
  explorerSayLit(found);
}

// How many games are lit, and what became of the rest. The map's own sentence says they
// are faded rather than gone, which is true of every control here but one: under *hide
// the rest* they are gone, and the line has to say so.
function explorerSayLit(found) {
  const said = document.getElementById('mapLit');
  if (!said) return;
  if (found && EXPLORER_ONLY) {
    said.textContent = `${found.size} of ${MAP_DATA.counts.nodes} games shown.`
      + ' The rest are hidden, and so is every line with an end among them.'
      + ' The legend still counts the whole library.';
  } else {
    mapSayLit(found);
  }
  const folded = EXPLORER_FOLDED.size;
  if (!folded) return;
  const held = mapGroups()
    .filter(group => EXPLORER_FOLDED.has(group.id))
    .reduce((sum, group) => sum + group.size, 0);
  said.textContent += ` ${folded === 1 ? 'One community is' : folded + ' communities are'}`
    + ' folded,'
    + ` so ${held} games are drawn as ${folded === 1 ? 'one dot' : folded + ' dots'}.`;
}

function explorerGoToFirstFound() {
  const found = explorerFound();
  if (!found) return;
  const index = MAP_DATA.nodes.findIndex(node => found.has(node.hash));
  if (index < 0) return;
  explorerOpenGame(index);
}

// ------------------------------------------------------------------------------ focus
//
// `docs/archive/plan-graph-map.md` S4: three ways of looking at part of the map without losing
// the rest of it. None of them moves a dot or works out a neighbour: a game's neighbours
// are the lines the library says are drawn to it, which is `connectedEdges` over the
// edges no switch or fold has taken off.

// The lines drawn now, as the library holds them. An arrow the switch hid is `off`, and a
// line a fold stands in for is `away`.
function explorerDrawnLines(part) {
  return part.connectedEdges().not('.off, .away');
}

// A pointer resting on a game lights it, its drawn lines and the games at their other
// ends, and fades everything else, until the pointer leaves. A pick that is held up is
// left alone: the tap is the stronger question, and a pointer crossing the map on the
// way to the panel must not undo it.
function explorerHover(dot) {
  if (!EXPLORER || MAP_PICK || dot.data('kind') !== 'node') return;
  const lines = explorerDrawnLines(dot).not('.aside');
  const near = dot.union(lines).union(lines.connectedNodes());
  EXPLORER_NEAR = new Set();
  EXPLORER.batch(() => {
    EXPLORER.elements().addClass('hushed');
    near.removeClass('hushed');
    near.nodes().forEach(part => {
      if (part.data('kind') !== 'node') return;
      EXPLORER_NEAR.add(MAP_DATA.nodes[part.data('at')].hash);
      EXPLORER.getElementById('r' + part.data('at')).removeClass('hushed');
    });
    dot.addClass('hovered');
  });
  explorerHullsFade();
  explorerPaintNames();
}

function explorerHushOff() {
  if (!EXPLORER || !EXPLORER_NEAR) return;
  EXPLORER_NEAR = null;
  EXPLORER.elements().removeClass('hushed hovered');
  explorerHullsFade();
  explorerPaintNames();
}

// The local view: one game and the games one or two steps out along the lines drawn now,
// with everything else set aside. It is repainted whenever the lines change, so turning
// the arrows on in a local view widens it along them.
function explorerPaintLocal() {
  if (!EXPLORER) return;
  EXPLORER.batch(() => {
    EXPLORER.nodes().removeClass('aside');
    if (!EXPLORER_LOCAL) return;
    let kept = EXPLORER.getElementById('n' + EXPLORER_LOCAL.at);
    for (let step = 0; step < EXPLORER_LOCAL.steps; step += 1) {
      kept = kept.union(explorerDrawnLines(kept).connectedNodes());
    }
    EXPLORER.nodes().forEach(part => {
      const dot = part.data('kind') === 'ring'
        ? EXPLORER.getElementById('n' + part.data('at')) : part;
      part.toggleClass('aside', !kept.contains(dot));
    });
  });
  explorerLocalSay();
  // Whatever changed what is drawn -- a cut, a fold, the local view, *hide the rest* --
  // ends here, so the outlines are drawn again over what is left.
  explorerPaintHulls();
}

// Into the local view of one game, a step further out, or back to the whole map. The
// view is fitted to what is shown, because a local view off the edge of the frame is a
// blank frame.
function explorerLocal(at, steps) {
  if (!EXPLORER) return;
  explorerHushOff();
  const node = at === null ? null : MAP_DATA.nodes[at];
  if (node) explorerUnfold(node.hash);
  EXPLORER_LOCAL = node
    ? { at: at, steps: Math.max(1, Math.min(EXPLORER_STEPS_OUT, steps)) } : null;
  explorerPaintLit();
  // The bar the local view writes over the drawing's top edge is room the fit cannot use.
  const bar = document.getElementById('exploreLocal');
  const room = EXPLORER_LOCAL && bar ? bar.offsetHeight + 9 : 0;
  const shown = EXPLORER.nodes().not('.aside, .away, .hidden, .ring');
  EXPLORER.fit(EXPLORER_LOCAL ? shown : undefined, EXPLORER_AIR + room);
}

// The key and the panel's press: into the local view of the game held up, or one step
// further out of the local view already on screen.
function explorerStepOut() {
  if (EXPLORER_LOCAL) return explorerLocal(EXPLORER_LOCAL.at, EXPLORER_LOCAL.steps + 1);
  if (MAP_PICK && MAP_PICK.kind === 'node') explorerLocal(MAP_PICK.at, 1);
}

// The bar over the drawing while a local view is on: what it shows, and the ways out.
function explorerLocalSay() {
  const bar = document.getElementById('exploreLocal');
  if (!bar) return;
  bar.hidden = !EXPLORER_LOCAL;
  if (!EXPLORER_LOCAL) { bar.textContent = ''; return; }
  const node = MAP_DATA.nodes[EXPLORER_LOCAL.at];
  const steps = EXPLORER_LOCAL.steps;
  const games = EXPLORER.nodes('[kind = "node"]').not('.aside, .away, .hidden').length;
  const button = (count, words) => `<button data-local-steps="${count}"${
    count === steps ? ' class="on" aria-pressed="true"' : ''}>${words}</button>`;
  bar.innerHTML = `<span class="small"><b>${esc(node.name)}</b> and the games ${
      steps === 1 ? 'one step' : steps + ' steps'} out along the lines drawn: ${
      games === 1 ? 'one game' : games + ' games'}.</span>
    <span class="row u-gap-4">${button(1, 'one step')}${
      button(EXPLORER_STEPS_OUT, 'two steps')}
      <button data-local-leave>whole map</button></span>`;
}

// Escape lets go of the innermost thing: the local view first, then the pick.
function explorerLetGo() {
  if (EXPLORER_LOCAL) return explorerLocal(null, 0);
  explorerPick(null);
}

// The find box's first match, centred and held while the reader types: marked, named
// first, and brought close enough to read its neighbours. Enter holds it up. A match
// folded into a group or set aside by the local view is not moved to, because a dot
// nobody can see cannot be centred.
function explorerHoldFirst() {
  if (!EXPLORER) return;
  const found = MAP_FIND.trim() ? explorerFound() : null;
  const index = found ? MAP_DATA.nodes.findIndex(node => found.has(node.hash)) : -1;
  const hash = index >= 0 ? MAP_DATA.nodes[index].hash : null;
  const moved = hash !== EXPLORER_FIRST;
  EXPLORER_FIRST = hash;
  EXPLORER.nodes('.held').removeClass('held');
  if (index >= 0) {
    const dot = EXPLORER.getElementById('n' + index);
    dot.addClass('held');
    if (moved && dot.visible()) {
      EXPLORER.stop();
      EXPLORER.animate({
        center: { eles: dot }, zoom: Math.max(EXPLORER.zoom(), EXPLORER_HOLD_ZOOM),
      }, { duration: 260 });
    }
  }
  explorerPaintNames();
}

// ------------------------------------------------------------------- holding one thing up
//
// The panel is the map's, so a tap fills in the state the map's panel functions read --
// `MAP_PICK` and, through `mapLit`, what that pick lights -- and calls them. What differs
// is the repaint, because the map's runs over an SVG this page has not got.

function explorerPick(pick) {
  // A pick is the stronger question, so the lighting a resting pointer did gives way.
  explorerHushOff();
  MAP_PICK = pick;
  MAP_ON = mapLit();
  explorerPaintPick();
  explorerPaintLit();
  explorerPaintCommunityChips();
  explorerPaintPanel();
}

// The key's chips: the one held up is on, whether as its community or as its folded dot,
// and a folded one says so.
function explorerPaintCommunityChips() {
  view.querySelectorAll('#mapCommunities [data-community]').forEach(chip => {
    const community = mapCommunityAt(Number(chip.dataset.community));
    const id = community && community.id;
    const group = MAP_PICK && MAP_PICK.kind === 'group' && mapGroupAt(MAP_PICK.at);
    const picked = MAP_PICK && MAP_PICK.kind === 'community'
      ? Number(chip.dataset.community) === MAP_PICK.at : !!group && group.id === id;
    chip.classList.toggle('on', !!picked);
    const folded = chip.querySelector('[data-folded]');
    if (folded) folded.hidden = !EXPLORER_FOLDED.has(id);
  });
}

// A community chip is a toggle: pressing the one already up lets it go. A folded
// community is held up as its dot, whose panel opens it again.
function explorerPickCommunity(index) {
  const community = mapCommunityAt(index);
  const at = community && EXPLORER_FOLDED.has(community.id)
    ? mapGroups().findIndex(group => group.id === community.id) : -1;
  const pick = at >= 0 ? { kind: 'group', at } : { kind: 'community', at: index };
  const same = MAP_PICK && MAP_PICK.kind === pick.kind && MAP_PICK.at === pick.at;
  explorerPick(same ? null : pick);
}

// The panel is the map's, and what the explorer has that the map has not is the site's
// own numbers. They are written under the map's panel rather than into it, so `#/map`
// goes on printing exactly what it printed.
//
// **The votes are printed beside the stars.** A mean of one vote draws as large a dot as
// a mean of twenty, and a reader who cannot see the count is reading a size that is not
// there (`docs/reference/no-dark-patterns.md`, test 7).
function explorerPaintPanel() {
  paintMapPanel();
  const panel = document.getElementById('mapPanel');
  if (!panel || !MAP_PICK || MAP_PICK.kind !== 'node') return;
  const node = MAP_DATA.nodes[MAP_PICK.at];
  const said = explorerPeopleWords(node && node.hash);
  if (!said) return;
  const line = document.createElement('div');
  line.className = 'muted small';
  line.style.marginTop = '10px';
  line.textContent = said;
  panel.appendChild(line);
}

// What people have done with one game, in one sentence. Every number in it is the
// server's own; what is chosen here is which of the two absences to say out loud, since
// *nobody has voted* and *nobody has played it* are different facts about a game.
function explorerPeopleWords(hash) {
  const record = hash && explorerPeopleAt(hash);
  if (!record) return '';
  const stars = record.stars === null || record.stars === undefined
    ? 'Nobody has scored this out of five yet.'
    : `${record.stars} out of five, from ${
        record.votes === 1 ? 'one vote' : record.votes + ' votes'}.`;
  const finished = record.played === 1 ? 'One game' : record.played + ' games';
  const played = record.played
    ? `${finished} finished here with a person in a seat.`
    : 'Not played here yet.';
  return `${stars} ${played}`;
}

function explorerPaintPick() {
  if (!EXPLORER) return;
  EXPLORER.elements().removeClass('picked');
  if (!MAP_PICK) return;
  const id = { node: 'n', line: 's', arrow: 'a', group: 'g' }[MAP_PICK.kind];
  if (!id) return;
  EXPLORER.getElementById(id + MAP_PICK.at).addClass('picked');
}

function explorerCentre(index) {
  if (!EXPLORER) return;
  const found = EXPLORER.getElementById('n' + index);
  if (found.length) EXPLORER.center(found);
}

// A game picked from the keyboard or from the find box: centred, held up, and the focus
// moved to the first thing its panel offers, which is *open this game*. A reader who
// pressed Enter on a name is asking for the game, not for a picture of it.
function explorerOpenGame(index) {
  const node = MAP_DATA.nodes[index];
  if (node) explorerUnfold(node.hash);
  explorerCentre(index);
  explorerPick({ kind: 'node', at: index });
  const open = document.querySelector('#mapPanel [data-open]');
  if (open) open.focus();
}

// ---------------------------------------------------------------------------- the keys
//
// Arrows pan, plus and minus zoom, 0 fits the whole library and Escape lets go, through
// the library's own `panBy`, `zoom` and `fit`. Enter on a game is written out rather than
// left to the button's own default, which is the map's rule for the same press.

function explorerKey(event) {
  const holder = event.target.closest('[data-game]');
  if (holder && (event.key === 'Enter' || event.key === ' ')) {
    event.preventDefault();
    return explorerOpenGame(Number(holder.dataset.game));
  }
  if (!EXPLORER) return;
  if (event.key === 'ArrowLeft') EXPLORER.panBy({ x: EXPLORER_PAN, y: 0 });
  else if (event.key === 'ArrowRight') EXPLORER.panBy({ x: -EXPLORER_PAN, y: 0 });
  else if (event.key === 'ArrowUp') EXPLORER.panBy({ x: 0, y: EXPLORER_PAN });
  else if (event.key === 'ArrowDown') EXPLORER.panBy({ x: 0, y: -EXPLORER_PAN });
  else if (event.key === '+' || event.key === '=') explorerZoom(EXPLORER_STEP);
  else if (event.key === '-' || event.key === '_') explorerZoom(1 / EXPLORER_STEP);
  else if (event.key === '0') EXPLORER.fit(undefined, EXPLORER_AIR);
  else if (event.key === 'Escape') explorerLetGo();
  // L is the local view: of the game held up, and then one step further out.
  else if (event.key === 'l' || event.key === 'L') explorerStepOut();
  else return;
  event.preventDefault();
}

// ------------------------------------------------------------------------- the wiring

function wireExplorer() {
  view.querySelectorAll('#mapCommunities [data-community]').forEach(chip => {
    chip.onclick = () => explorerPickCommunity(Number(chip.dataset.community));
  });
  const arrows = document.getElementById('exploreArrows');
  if (arrows) arrows.onchange = () => explorerSetArrows(arrows.checked);
  const paint = document.getElementById('explorePaint');
  if (paint) paint.onchange = () => explorerSetPaint(paint.value);
  const hulls = document.getElementById('exploreHulls');
  if (hulls) hulls.onchange = () => explorerSetHulls(hulls.checked);
  const local = document.getElementById('exploreLocal');
  if (local) {
    local.onclick = event => {
      const out = event.target.closest('[data-local-steps]');
      if (out && EXPLORER_LOCAL) {
        return explorerLocal(EXPLORER_LOCAL.at, Number(out.dataset.localSteps));
      }
      if (event.target.closest('[data-local-leave]')) explorerLocal(null, 0);
    };
  }
  const reset = document.getElementById('exploreReset');
  if (reset) reset.onclick = () => explorerReset();
  // On a phone the whole sheet of controls folds to its one line, so the drawing is on the
  // first screen (archive/plan-visual.md §1.1: the controls took it); one tap opens them.
  const controls = document.getElementById('mapControls');
  if (controls) controls.open = window.innerWidth > 720;
  const find = document.getElementById('mapFind');
  if (find) {
    find.oninput = () => {
      MAP_FIND = find.value;
      // Two ways of fading the map at once read as one muddle, and the newer question is
      // the one being typed.
      if (MAP_FIND.trim() && MAP_PICK) explorerPick(null);
      else explorerPaintLit();
      explorerHoldFirst();
    };
    find.onkeydown = event => {
      if (event.key === 'Enter') { event.preventDefault(); explorerGoToFirstFound(); }
    };
  }
  const show = document.getElementById('mapShow');
  if (show) {
    show.onchange = () => {
      MAP_SHOW = show.value;
      if (MAP_SHOW && MAP_PICK) return explorerPick(null);
      explorerPaintLit();
    };
  }
  const sized = document.getElementById('exploreSize');
  if (sized) {
    sized.onchange = () => {
      EXPLORER_SIZE = sized.value;
      explorerSizeSay();
      explorerPaintSizes();
    };
  }
  explorerSizeSay();
  const kept = document.getElementById('exploreKeep');
  if (kept) {
    kept.onchange = () => {
      EXPLORER_KEEP = kept.value;
      // The same rule the score menu works under: a pick and a filter fading the map at
      // once read as one muddle, and the filter is the newer question.
      if (EXPLORER_KEEP && MAP_PICK) return explorerPick(null);
      explorerPaintLit();
    };
  }
  const only = document.getElementById('exploreOnly');
  if (only) {
    only.onchange = () => { EXPLORER_ONLY = only.checked; explorerPaintLit(); };
  }
  const keys = document.getElementById('exploreKeys');
  if (keys) {
    keys.onclick = event => {
      const holder = event.target.closest('[data-game]');
      if (holder) explorerOpenGame(Number(holder.dataset.game));
    };
    // A game Tab lands on is brought to the middle. Tabbing to something off screen and
    // having nothing happen is the way a keyboard reader loses a map.
    keys.addEventListener('focusin', event => {
      const holder = event.target.closest('[data-game]');
      if (holder) explorerCentre(Number(holder.dataset.game));
    });
  }
  const frame = document.getElementById('exploreFrame');
  if (frame) frame.addEventListener('keydown', explorerKey);
  wireExplorerPanel();
  explorerWatchTheme();
  explorerPaintLines();
  explorerFoldPaint();
}

// A colour here is the colour of the theme that was on when it was read, so a theme
// change is a repaint: every element's colours taken from the payload again and the
// stylesheet built afresh. Positions are left where they are, because they are not this
// function's to move. Wired once for the page's life, because the explorer is drawn anew
// on every visit and the handler is not.
function explorerWatchTheme() {
  if (explorerWatchTheme.wired) return;
  explorerWatchTheme.wired = true;
  window.matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change', () => explorerRepaintTheme());
  // The footer's own switch (`shell.js`), which no media query sees.
  document.addEventListener('themechange', () => explorerRepaintTheme());
}

function explorerRepaintTheme() {
  if (!EXPLORER || !MAP_DATA) return;
  EXPLORER.batch(() => {
    explorerElements().forEach(part => {
      const held = EXPLORER.getElementById(part.data.id);
      if (!held.length) return;
      ['fill', 'edge', 'colour'].forEach(key => {
        if (part.data[key]) held.data(key, part.data[key]);
      });
    });
  });
  // A folded dot's lines are not in the elements above -- they are built from whatever is
  // folded -- so they are built again in the colours of the theme that is on now.
  explorerFoldPaint();
  EXPLORER.style(explorerStyle());
}

// One handler for the panel, which is redrawn on every pick, so a handler per button
// would be rewired each time. A press that holds something else up goes through this
// module, because a pick is a class on a drawn element; everything else -- opening a
// game, asking for the design steps, taking one -- is `page/graph.js`'s
// own function, called with what the press named.
function wireExplorerPanel() {
  const panel = document.getElementById('mapPanel');
  if (!panel) return;
  panel.onclick = event => {
    const open = event.target.closest('[data-open]');
    if (open) {
      pending('Loading the game…');
      return go('#/game/' + open.dataset.open);
    }
    const asking = event.target.closest('[data-try]');
    if (asking) return mapAskSteps(asking.dataset.try);
    const step = event.target.closest('[data-step]');
    if (step) {
      return mapTakeStep(step.dataset.from, step.dataset.step, step.dataset.seed);
    }
    const row = event.target.closest('[data-line-row]');
    if (row) return explorerPick({ kind: 'line', at: Number(row.dataset.lineRow) });
    const local = event.target.closest('[data-local]');
    if (local) return explorerLocal(Number(local.dataset.local), 1);
    const community = event.target.closest('[data-community]');
    if (community) return explorerPickCommunity(Number(community.dataset.community));
    const node = event.target.closest('[data-node-row]');
    if (node) {
      const index = MAP_DATA.nodes.findIndex(one => one.hash === node.dataset.nodeRow);
      if (index < 0) return;
      // A game named in a folded group's panel: the group opens and that game is picked,
      // which is what pressing its name there is asking for.
      explorerUnfold(node.dataset.nodeRow);
      explorerCentre(index);
      return explorerPick({ kind: 'node', at: index });
    }
    const opened = event.target.closest('[data-open-group], [data-fold-group]');
    if (opened) {
      const id = opened.dataset.openGroup || opened.dataset.foldGroup;
      const at = mapGroups().findIndex(one => one.id === id);
      if (at >= 0) explorerFoldToggle(at);
      return;
    }
  };
}
