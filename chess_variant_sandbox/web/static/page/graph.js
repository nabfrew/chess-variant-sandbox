// The library as a map, in words: the legend, the panels, and what they read.
//
// `docs/archive/plan-game-graph.md` G3 to G7 built the map's words; `docs/plan-map.md` M4
// put the new map under them. The second view of the library -- *list* beside *map* --
// drawn from one call to `GET /api/graph`. The drawing is `page/explorer.js`'s and is what
// `#/map` opens; what lives here is everything the map says in words, and the readings of
// the payload the drawing and the panels share.
//
// **This module places nothing and computes nothing.** Every coordinate is the server's and
// no coordinate is written here at all. It works out no likeness between two games,
// compares no rules, reads no spec and asks the engine nothing: which games are alike,
// what they share and what differs between them, which communities they form and which
// games are unusual all arrive in the payload, already decided.
//
// Three things the map has to answer that the payload deliberately leaves open:
//
// **A line has to say which of two things it is.** A line in a community's colour, or in
// grey between two communities, joins two games that are each among the other's nearest,
// and its width is how alike they are; a grey line with an arrowhead is *this game was made
// from that one*. They are different claims, and the legend carries a swatch of each.
//
// **A name at every dot is a smear.** Ninety names at the width of the whole map overlap
// several times over, so `mapLabelOrder` ranks them -- whatever is held up first, then each
// community's centre, then the busiest games -- and the drawing paints down that order
// until a name would land on one already there. Zooming in makes room for the rest.
//
// **A community is a place, or two.** The server's communities are a partition -- every
// game in one or none -- so a community can be outlined on the drawing, lit from its chip
// and folded into one dot, and a game is never in two of them. The picture keeps games
// alike near each other, so a community whose halves share little but one thing can be
// drawn in two places; the server says so (`parts`) and each place gets its own outline.
//
// Two more the score brought with it (G4):
//
// **A fill needs two ends, and nought to one is not them.** Sixty-eight of the seventy games
// score between 0.17 and 0.82 with the median at 0.75, so a fill run over the whole unit
// paints the library one colour. The ends are the payload's (`scale`), the legend prints
// both numbers, and a game with no current score keeps the hollow dot it had. A game scored
// under other weights is hollow too, because two such scores never share a scale.
//
// **A filter lights, it never hides.** The listing's filters narrow a list of cards, where a
// card that goes takes nothing with it. Here a node that went would leave its lines hanging
// off nothing and would take a reader's landmarks with it, so the map fades what does not
// match and moves nothing -- the same thing a pick already does, and the same thing the find
// box already did with a name.
//
// And one the remix brought (G6):
//
// **A game a reader makes from the map is made by the server.** *And now try* ends in the
// editor holding a draft, and does not work out what that draft is: which small changes to
// the rules apply to a game is the designer's answer, asked of `GET /api/design/<hash>`,
// which is the route *make your own from this* already asks.

// ------------------------------------------------- the map, and what a reader asked of it
//
// One payload and the state a reader has put it in. `page/explorer.js` draws from these
// and writes back to them, because one library read two ways would be two libraries.

let MAP_DATA = null,   // the payload, exactly as `/api/graph` sent it
    MAP_AT = {},       // hash -> node, which is the join every line needs
    // What is held up: {kind: 'node'|'line'|'arrow'|'community'|'group', at: <index>}.
    // One at a time -- holding two things up is holding nothing up.
    MAP_PICK = null,
    MAP_ON = null,     // the hashes and line indexes that pick lights, as three Sets
    MAP_FIND = '',     // what is typed in the find box, which lights games by name
    MAP_SHOW = '',     // which measurements are lit, in the listing's own five states
    // Whether the arrows from a game to one made from it are drawn. Off when the map
    // opens: the lines of likeness are the map.
    MAP_ARROWS = false,
    // What a dot's fill says: 'community', the colour of the community it is drawn in,
    // or 'ramp', its score on the library's own ramp.
    MAP_PAINT = 'community',
    // What the server has said about remixing one game, by hash: {state, steps, why}.
    // Kept across repaints of the panel, because the panel is redrawn on every pick and
    // asking again on each would be a request per click.
    MAP_TRY = {};

// The two ways of reading the same library, as one control. Drawn on both views so the
// way back is where the way in was.
function viewSwitchHtml(here) {
  return `<div class="viewSwitch" role="group" aria-label="how to show the library">
    <button data-view="list"${here === 'list' ? ' class="on" aria-current="true"' : ''}>list</button>
    <button data-view="map"${here === 'map' ? ' class="on" aria-current="true"' : ''}>map</button>
  </div>`;
}

function wireViewSwitch() {
  view.querySelectorAll('[data-view]').forEach(button => {
    button.onclick = () => {
      if (button.classList.contains('on')) return;
      if (button.dataset.view === 'map') pending('Drawing the map…');
      go(button.dataset.view === 'map' ? '#/map' : '#/');
    };
  });
}

function mapSummary() {
  const counts = MAP_DATA.counts;
  return `${counts.nodes} games in ${counts.communities} communities, and ${
    counts.unusual} unusual ones`;
}

// ------------------------------------------------------------------------ the lines
//
// `docs/plan-map.md` M4. Every pair of games each among the other's nearest is a row in
// `neighbours`, carrying how alike the two are (`likeness`, which is the line's width),
// the rarest things both have (`shares`), and what differs between them in rules, which
// is `spec/relate.py`'s reading. There is no cut to choose: which lines exist is the
// server's, and every one of them is drawn.

function mapLines() {
  return (MAP_DATA && MAP_DATA.neighbours) || [];
}

// The index of the line between two games, or -1 where none joins them.
function mapLineBetween(one, other) {
  return mapLines().findIndex(line =>
    (line.source === one && line.target === other)
    || (line.source === other && line.target === one));
}

// ----------------------------------------------------------------- the communities
//
// The server's partition of the games that have a community, with a name for each, its
// centre and where it is drawn. Every community of three or more gets a colour of its own,
// in the payload's order, while there are colours to give; a pair is already visible as a
// pair, and eight is about as many colours as a reader tells apart. A game in no coloured
// community keeps the neutral dot.

const MAP_COLOURED = 3;
const MAP_COLOURS = 8;

function mapCommunities() {
  return (MAP_DATA && MAP_DATA.communities) || [];
}

function mapCommunityAt(index) {
  return mapCommunities()[index] || null;
}

function mapCommunityOf(node) {
  const at = node ? mapCommunities().findIndex(one => one.id === node.community) : -1;
  return at >= 0 ? { community: mapCommunities()[at], index: at } : null;
}

// A community's colour as a custom property, or nothing for the neutral one. Which
// colour follows the community's place in the payload, never what is lit or hidden, so
// a filter never repaints the communities it leaves.
function mapCommunityColour(id) {
  const coloured = mapCommunities().filter(one => one.size >= MAP_COLOURED);
  const at = coloured.findIndex(one => one.id === id);
  return id && at >= 0 && at < MAP_COLOURS ? `var(--community-${at + 1})` : '';
}

// A line's colour: its community's where both ends are in the same one, and the neutral
// colour where they are not.
function mapLineColour(line) {
  const from = MAP_AT[line.source], to = MAP_AT[line.target];
  const shared = from && to && from.community && from.community === to.community;
  return (shared && mapCommunityColour(from.community)) || 'var(--muted)';
}

// ------------------------------------------------------------------------- the legend
//
// What the dots, the lines and the colours mean, in words. The communities are the key,
// and the one list of them; the score ramp says what a fill means only while the fill is
// the score; the two kinds of line are a swatch each.

function mapLegendHtml() {
  return `<div class="mapLegend">
    ${mapCommunitiesHtml()}
    <div id="mapScale"${MAP_PAINT === 'ramp' ? '' : ' hidden'}>${mapScaleHtml()}</div>
    <div class="row muted small" style="gap:4px 14px">
      <span class="row" style="gap:6px; flex-wrap:nowrap">${mapLineSwatch('line')}<span>a line: each is
        among the other's ${MAP_DATA.mutual} nearest, thicker the more alike${
        info('alike')}</span></span>
      <span class="row" style="gap:6px; flex-wrap:nowrap"${MAP_ARROWS ? '' : ' hidden'}
        id="mapArrowsKey">${mapLineSwatch('lineage')}<span>made from <span
        class="muted">${MAP_DATA.counts.lineage}</span></span></span>
    </div>
  </div>`;
}

// One chip per community, in the payload's order, each with its colour, its name and
// how many games it holds. Pressing one lights that community and fades the rest; the
// names are the server's. This is the one list of them: folding one is in its panel
// (Q801), and a folded one's chip says so.
function mapCommunitiesHtml() {
  const communities = mapCommunities();
  if (!communities.length) return '';
  const chips = communities.map((community, index) => {
    const colour = mapCommunityColour(community.id);
    return `<button class="chip${MAP_PICK && MAP_PICK.kind === 'community'
      && MAP_PICK.at === index ? ' on' : ''}" data-community="${index}"
      title="${esc(community.names.join(', '))}">${colour
        ? `<span class="dot" style="background:${esc(colour)}"></span>`
        : mapDotSwatch('var(--panel)', 'var(--muted)')}${esc(community.name)}
      <span class="muted">${community.size}</span><span class="muted" data-folded
        hidden>&middot; folded</span></button>`;
  }).join('');
  // Folded on a phone, where sixteen chips are a screen of their own above the map.
  return `<details id="mapCommunitiesBox"${window.innerWidth > 720 ? ' open' : ''}>
    <summary class="muted small">${communities.length} communities${info('community')},
      drawn together and in one colour</summary>
    <div class="muted small" style="margin:6px 0">Each is named by what marks it out from
      the rest of the library. Click one to light it; its panel folds it into one dot. A
      hollow dot is a game in a pair, or an unusual game in none.</div>
    <div class="chips" id="mapCommunities">${chips}</div></details>`;
}

// What a dot's fill says, which is the one thing on the map a colour means that is not a
// community. Both ends of the ramp are printed as numbers: the scale is the library's
// own range rather than nought to one, so a colour read against an imagined scale would
// be read wrong. Five stepped dots rather than a gradient, because a dot is a shape the
// stylesheet already has and this legend is a key rather than a picture.
function mapScaleHtml() {
  const scale = MAP_DATA.scale;
  if (!scale) return '';
  const steps = [0, 0.25, 0.5, 0.75, 1]
    .map(at => mapDotSwatch(mapFillAt(at), 'var(--line)')).join('');
  // No separator between the four: they wrap onto two or three lines on a phone, and a
  // dot left dangling at the end of a wrapped line is the untidiest way to go on.
  return `<div class="row muted small" style="gap:4px 14px">
    <span class="row" style="gap:3px">how it scored${info('index')}
      <span>${esc(fmt(scale.low))}</span>${steps}<span>${esc(fmt(scale.high))}</span></span>
    <span>${mapDotSwatch('var(--panel)', 'var(--muted)')}no score, or out of
      date${info('stale')}</span>
    <span>${mapRuledSwatch()}scored, then ruled out${info('gated')}</span>
    <span>${scale.scored} of ${MAP_DATA.counts.nodes} games carry a score</span>
  </div>`;
}

// A dot the size the legend wants it, drawn rather than styled: the stylesheet's own dot
// is a chip's, and one outside a chip has no size at all.
function mapDotSwatch(fill, stroke) {
  return `<svg class="swatch" width="13" height="13" aria-hidden="true"><circle cx="6.5"
    cy="6.5" r="5.5" fill="${esc(fill)}" stroke="${esc(stroke)}"></circle></svg>`;
}

// The mark a game the gates ruled out carries, drawn the way the map draws it: a ring
// inside the dot rather than round it, so two neighbours' marks cannot touch.
function mapRuledSwatch() {
  return `<svg class="swatch" width="13" height="13" aria-hidden="true"><circle cx="6.5"
    cy="6.5" r="5.5" fill="var(--panel)" stroke="var(--muted)"></circle><circle cx="6.5"
    cy="6.5" r="3" fill="none" stroke="var(--bad)" stroke-width="1.4"></circle></svg>`;
}

// One point on the ramp, as a colour. The two ends are the theme's own -- the page's mark
// colour mixed into the panel a dot sits on -- so the same score reads the same way under
// either theme and nothing is composited through a layer of its own. The floor keeps the
// lowest score a shade rather than nothing at all, because nothing at all is what a game
// with no score looks like.
function mapFillAt(at) {
  const share = Math.round(12 + 88 * Math.max(0, Math.min(1, at)));
  return `color-mix(in oklab, var(--meter) ${share}%, var(--panel))`;
}

// The two kinds of line, as the legend draws them: a line of likeness, solid, and an
// arrow for *made from*, grey with its head.
function mapLineSwatch(type) {
  const arrow = type === 'lineage'
    ? '<path d="M14,1.2 L20,5 L14,8.8 z" fill="var(--muted)"></path>' : '';
  const colour = type === 'lineage' ? 'var(--muted)' : 'var(--ink)';
  return `<svg class="swatch" width="22" height="10" aria-hidden="true"><line x1="1" y1="5"
    x2="${type === 'lineage' ? 15 : 21}" y2="5" stroke="${colour}" stroke-width="${
      type === 'lineage' ? 1.6 : 2.6}" stroke-linecap="round"></line>${arrow}</svg>`;
}

// A kind of rule's colour, with grey behind it: a kind the schema grows before the
// stylesheet does draws in the fallback rather than not at all.
function mapKindColour(kind) {
  return `var(--kind-${kind}, var(--muted))`;
}

// What one game's dot is filled with, or nothing -- and nothing is the hollow dot the
// stylesheet draws, which is what an unscored game and an out-of-date one both are. A
// score under other weights is hollow too: two scores taken under different scoring rules
// never share a leaderboard, so they must not share a ramp either.
function mapScoreFill(node) {
  const scale = MAP_DATA.scale;
  const score = node.score;
  if (!scale || !score || score.index === null || score.index === undefined) return '';
  if (score.weights !== scale.weights) return '';
  const span = scale.high - scale.low;
  return mapFillAt(span > 0 ? (score.index - scale.low) / span : 1);
}

// The score in a sentence, for the panel and for the list of games a keyboard reader
// tabs through beside the drawing. The number is never
// without its run count, which is `page/core.js`'s rule and the same phrase a card uses.
function mapScoreWords(node) {
  const score = node.score;
  if (!score || score.index === null || score.index === undefined) {
    return node.staleness && !node.staleness.current ? 'score out of date' : 'no score yet';
  }
  const rested = runsPhrase(score.seeds);
  return `scores ${fmt(score.index)}${rested ? ', as the ' + rested : ''}${
    score.viable === false ? ', and ruled out' : ''}`;
}

// How many rules apart one line's two games are, in the words the panels use.
function mapRulesApart(line) {
  if (!line.units) return 'the same rules, written two ways';
  return line.units === 1 ? 'one rule apart' : `${line.units} rules apart`;
}

// --------------------------------------------------------------------------- the names
//
// Which names a reader can have, in the order they are worth having. The drawing works
// out how many fit and paints them; what is decided here is which one wins when two want
// the same piece of screen.

// Whatever the drawing asks to name first (a hovered game and its neighbours, the find
// box's first match), then whichever games are held up or found, then each community's
// centre, biggest community first, so the whole-map view names every community; then the
// busiest, then by name so that two games of one degree do not swap their names between
// two paints.
function mapLabelOrder(first) {
  const order = MAP_DATA.nodes.map((node, index) => index);
  const found = mapFound();
  const lit = node => (
    (MAP_ON && MAP_ON.nodes.has(node.hash)) || (found && found.has(node.hash)) ? 1 : 0);
  const ahead = node => (first && first.has(node.hash) ? 1 : 0);
  const centres = {};
  mapCommunities().forEach((community, index) => { centres[community.centre] = index + 1; });
  const centre = node => centres[node.hash] || Infinity;
  return order.sort((left, right) => {
    const a = MAP_DATA.nodes[left], b = MAP_DATA.nodes[right];
    return ahead(b) - ahead(a) || lit(b) - lit(a)
      || (centre(a) === centre(b) ? 0 : centre(a) < centre(b) ? -1 : 1)
      || b.degree - a.degree || a.name.localeCompare(b.name);
  });
}

function mapOverlaps(one, other) {
  return one.x < other.x + other.w && other.x < one.x + one.w
    && one.y < other.y + other.h && other.y < one.y + one.h;
}

// -------------------------------------------------------- finding a game, and filtering
//
// Ninety dots and no way to ask *where is this one* would make the map a puzzle for
// anybody who arrived knowing which game they wanted. Typing lights every name that
// matches and fades the rest. Enter goes to the first match.
//
// **A filter lights; it never hides.** The listing narrows to the cards that match and a
// card that goes takes nothing with it. A node that went would leave its lines hanging
// off nothing and would take away the landmarks the reader is steering by, so the same
// fade the find box uses answers the filter as well. Nothing moves, no line goes, and the
// legend's counts are still counts of what is drawn -- how many games are lit is said
// separately, under the controls.
//
// **The filter is the five states of the listing's own score menu**, because those are
// the questions a node can answer out of the payload. The rest of the listing's filters
// read fields only the listing has, and the map asks for one payload and nothing else.

function mapMatches(node) {
  const score = node.score;
  const stale = !!(node.staleness && !node.staleness.current);
  if (MAP_SHOW === 'yes' && !score) return false;
  if (MAP_SHOW === 'viable' && !(score && score.viable === true)) return false;
  if (MAP_SHOW === 'stale' && !stale) return false;
  if (MAP_SHOW === 'no' && (score || stale)) return false;
  return true;
}

function mapFound() {
  const text = MAP_FIND.trim().toLowerCase();
  if (!text && !MAP_SHOW) return null;
  return new Set(MAP_DATA.nodes
    .filter(node => mapMatches(node)
      && (!text || node.names.some(name => name.toLowerCase().includes(text))))
    .map(node => node.hash));
}

// How many games are lit, said once and only while something is narrowing them. The
// sentence has to say that the rest are still there, because on a list the same control
// would have taken them away.
function mapSayLit(found) {
  const said = document.getElementById('mapLit');
  if (!said) return;
  if (!found) { said.textContent = ''; return; }
  said.textContent = `${found.size} of ${MAP_DATA.counts.nodes} games lit. `
    + 'The rest are faded rather than gone, so every line is still where it was.';
}

// ----------------------------------------------------------------- holding one thing up

// The nodes and the lines one pick lights, as three sets. Everything else on the map is
// faded rather than hidden, so the shape a reader is looking at stays where it was.
function mapLit() {
  const nodes = new Set(), lines = new Set(), arrows = new Set();
  if (!MAP_PICK) return null;
  // The lines with an end among `inside`, and their ends; or, with `both`, only those
  // with both ends among them.
  const alike = (inside, both) => mapLines().forEach((line, index) => {
    const ends = [inside.has(line.source), inside.has(line.target)];
    if (both ? !(ends[0] && ends[1]) : !(ends[0] || ends[1])) return;
    lines.add(index);
    nodes.add(line.source);
    nodes.add(line.target);
  });
  if (MAP_PICK.kind === 'node') {
    const node = MAP_DATA.nodes[MAP_PICK.at];
    if (!node) return null;
    nodes.add(node.hash);
    alike(new Set([node.hash]), false);
    // An arrow that is not drawn lights nothing: its far end would be lit for a line the
    // reader cannot see.
    MAP_DATA.lineage.forEach((arrow, index) => {
      if (arrow.source !== node.hash && arrow.target !== node.hash) return;
      if (mapArrowHidden()) return;
      arrows.add(index);
      nodes.add(arrow.source);
      nodes.add(arrow.target);
    });
  } else if (MAP_PICK.kind === 'line') {
    const line = mapLines()[MAP_PICK.at];
    if (!line) return null;
    lines.add(MAP_PICK.at);
    nodes.add(line.source);
    nodes.add(line.target);
  } else if (MAP_PICK.kind === 'arrow') {
    const arrow = MAP_DATA.lineage[MAP_PICK.at];
    if (!arrow) return null;
    arrows.add(MAP_PICK.at);
    nodes.add(arrow.source);
    nodes.add(arrow.target);
  } else if (MAP_PICK.kind === 'group') {
    // A folded group stands for its members, so it lights what they light: themselves,
    // and every line either end of which is one of them.
    const group = mapGroupAt(MAP_PICK.at);
    if (!group) return null;
    const inside = new Set(group.members);
    inside.forEach(hash => nodes.add(hash));
    alike(inside, false);
  } else if (MAP_PICK.kind === 'community') {
    // A community lights its own games and the lines drawn between them.
    const community = mapCommunityAt(MAP_PICK.at);
    if (!community) return null;
    const inside = new Set(community.members);
    inside.forEach(hash => nodes.add(hash));
    alike(inside, true);
  }
  return { nodes, lines, arrows };
}

// Whether the arrows are off: one switch, off at opening.
function mapArrowHidden() {
  return !MAP_ARROWS;
}

// --------------------------------------------------------------------------- the panel

function paintMapPanel() {
  const panel = document.getElementById('mapPanel');
  if (!panel) return;
  panel.innerHTML = mapPanelHtml();
  // Stacked, the panel is under the map and a pick would otherwise answer off screen.
  if (MAP_PICK && window.innerWidth <= 900) {
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function mapPanelHtml() {
  if (MAP_PICK && MAP_PICK.kind === 'node') return mapNodePanel(MAP_DATA.nodes[MAP_PICK.at]);
  if (MAP_PICK && MAP_PICK.kind === 'line') return mapLinePanel(mapLines()[MAP_PICK.at]);
  if (MAP_PICK && MAP_PICK.kind === 'arrow') return mapArrowPanel(MAP_DATA.lineage[MAP_PICK.at]);
  if (MAP_PICK && MAP_PICK.kind === 'group') return mapGroupPanel(mapGroupAt(MAP_PICK.at));
  if (MAP_PICK && MAP_PICK.kind === 'community') {
    return mapCommunityPanel(mapCommunityAt(MAP_PICK.at));
  }
  return mapNothingPanel();
}

function mapNothingPanel() {
  const counts = MAP_DATA.counts;
  // The unusual games, furthest from everything first: a finding, not a leftover row.
  const unusual = (MAP_DATA.unusual || []).map(one => MAP_AT[one.hash]).filter(Boolean);
  return `<h3 style="margin-top:0">The library, laid out</h3>
    <div class="small">${counts.nodes} games, each drawn near the games most like it. A
      line joins two games when each is among the other's ${MAP_DATA.mutual} nearest, and
      the thicker the line the more alike the two${info('alike')}. A community${
      info('community')} is a group of games more like each other than like the rest,
      drawn together in one colour and named by what marks it out.</div>
    ${unusual.length ? `<div class="muted small" style="margin-top:10px">${unusual.length}
      unusual games are in no community: nothing in the library is close enough to them.
      The furthest from everything first:</div>
    <div class="chips" style="margin-top:6px">${unusual.map(node => `<button class="chip"
      data-node-row="${esc(node.hash)}">${esc(node.name)}</button>`).join('')}</div>`
      : ''}
    ${MAP_DATA.scale ? `<div class="muted small" style="margin-top:10px">Coloured by
      score, a game's dot is filled by what it scored, deeper for a higher score, and left
      hollow where there is no ${ACCOUNTS ? 'current score'
        : 'score this checkout can read'}. The ring inside a dot marks a
      game the checks ruled out.</div>` : ''}
    <div class="muted small" style="margin-top:10px">Point at a game to light it and its
      neighbours, click it for what it is and the games like it, or a line for what the
      two share and what differs. Drag to move the map, scroll or pinch to zoom, and Tab to
      walk the games from the keyboard.</div>
    ${counts.twins ? `<div class="muted small" style="margin-top:10px">${counts.twins === 1
      ? 'Two of the games are the same rules written two ways, shown as one dot.'
      : `${counts.twins} pairs of games are the same rules written two ways, each pair
      shown as one dot.`}</div>` : ''}`;
}

function mapNodePanel(node) {
  if (!node) return mapNothingPanel();
  const score = node.score || {};
  // The card's own three fields, under the card's own spellings, so the map and the
  // listing cannot come to describe one measurement two ways.
  const tag = scoreTag({ ...score, staleness: node.staleness });
  const split = splitSentences(score.runVerdict, score.runs);
  const home = mapCommunityOf(node);
  return `<h2 style="margin:0 0 6px">${esc(node.name)}</h2>
    <div class="row pills">${tag}</div>
    ${exhibitHtml(node.exhibit)}
    ${node.names.length > 1 ? `<div class="muted small">also listed as ${
      esc(node.names.slice(1).join(', '))}: the same rules, written two ways.</div>` : ''}
    <div class="row muted small" style="gap:8px">
      <span>${esc(node.board)}</span><span>&middot;</span>
      <span>${node.degree === 1 ? '1 line' : node.degree + ' lines'}</span>
    </div>
    ${split.length ? `<div class="muted small" style="margin-top:6px">${
      esc(split.join(' '))}</div>` : ''}
    <div class="row" style="margin-top:10px">
      <button class="primary" data-open="${esc(node.hash)}">open this game</button>
      <button data-try="${esc(node.hash)}">and now try&hellip;</button>
      <button data-local="${MAP_DATA.nodes.indexOf(node)}" title="Hide every game that is
        not this one or a neighbour; press L to step one further out.">only this and its
        neighbours</button>
    </div>
    ${mapTryHtml(node)}
    ${home ? `<div class="muted small" style="margin-top:10px">In the community
      <button class="chip" data-community="${home.index}">${
        esc(home.community.name)}</button></div>`
      : `<div class="muted small" style="margin-top:10px">Unusual: in no community, because
        nothing in the library is close enough to it.</div>`}
    ${mapLikeHtml(node)}`;
}

// The games like this one, nearest first, each with the rarest things the two share. A
// game joined to this one by a line -- each among the other's nearest -- holds that line
// up when pressed, and says what the two scores make of it; any other holds that game up.
// The order, the words and which are joined are the server's.
function mapLikeHtml(node) {
  const like = (node.like || []).filter(row => MAP_AT[row.hash]);
  if (!like.length) return '';
  const rows = like.map(row => ({ row, at: row.line ? mapLineBetween(node.hash, row.hash) : -1 }));
  // Said once for the whole list rather than under every row, which is the game page's
  // rule: where nothing here can be read, five rows of the same apology is the loudest
  // thing in the panel.
  const scored = rows.some(({ at }) => at >= 0 && mapLines()[at].gap);
  return `<div class="muted small" style="margin-top:12px">Games like this one, most alike
      first${info('alike')}</div>
    ${scored || !rows.some(({ at }) => at >= 0) ? '' : `<div class="muted small"
      style="margin-top:6px">No scores to compare yet. A line says what the difference did
      to the score once both games ${ACCOUNTS ? 'have a current score'
        : 'carry a measurement this checkout can read'}.</div>`}
    <div class="links">${rows.map(({ row, at }) => {
      const other = MAP_AT[row.hash];
      const line = at >= 0 ? mapLines()[at] : null;
      return `<button class="linkRow" ${line ? `data-line-row="${at}"`
        : `data-node-row="${esc(row.hash)}"`}>
      <div class="row" style="gap:8px; align-items:baseline">
        <b>${esc(other.name)}</b><span class="muted small">${esc(fmt(row.likeness))} alike${
          line ? ', a line' : ''}</span>
      </div>
      ${row.shares.length ? `<div class="small muted">both have ${
        esc(row.shares.join('; '))}</div>` : ''}
      ${line && scored ? gapLine(mapGapFrom(line, node.hash)) : ''}
    </button>`;
    }).join('')}</div>`;
}

// One line: how alike its two games are, what they share and what differs between them,
// and what the two scores make of it. Every word is the payload's.
function mapLinePanel(line) {
  const from = line && MAP_AT[line.source], to = line && MAP_AT[line.target];
  if (!from || !to) return mapNothingPanel();
  const rules = (line.differences || []).length ? `<div class="links">${
    line.differences.map(difference => `
      <div class="linkRow">
        <span class="chip" style="cursor:default"><span class="dot"
          style="background:${esc(mapKindColour(difference.kind))}"></span>${
          esc(mapKindWords(difference.kind))}</span>
        <div class="small" style="margin-top:4px">${esc(difference.sentence)}</div>
      </div>`).join('')}</div>` : '';
  return `<h3 style="margin-top:0">How alike</h3>
    <div class="row" style="gap:8px">
      <button class="crumb" data-node-row="${esc(line.source)}"
        style="color:var(--ink); font-weight:600">${esc(from.name)}</button>
      <span class="muted small">and</span>
      <button class="crumb" data-node-row="${esc(line.target)}"
        style="color:var(--ink); font-weight:600">${esc(to.name)}</button>
    </div>
    <div class="small" style="margin-top:8px">${esc(fmt(line.likeness))} alike, where 1 is
      the same game and 0 means they share nothing rare${info('alike')}. Each is among the
      other's ${MAP_DATA.mutual} nearest, which is why a line joins them.</div>
    ${line.shares.length ? `<div class="small" style="margin-top:8px"><span
      class="muted">Both have:</span> ${esc(line.shares.join('; '))}</div>` : ''}
    <div class="small" style="margin-top:8px"><span class="muted">What differs:</span>
      ${esc(mapRulesApart(line))}${(line.differences || []).length ? ''
        : line.units ? ', too many to list one by one' : ''}.</div>
    ${rules}
    ${mapGapHtml(line, from, to)}
    <div class="row" style="margin-top:10px">
      <button data-open="${esc(line.source)}">open ${esc(from.name)}</button>
      <button data-open="${esc(line.target)}">open ${esc(to.name)}</button>
    </div>`;
}

// One community: the server's name for it, what marks it out in the server's words where
// something does, the game most like the rest of it, where it is drawn, and every member
// as a press.
function mapCommunityPanel(community) {
  if (!community) return mapNothingPanel();
  const centre = MAP_AT[community.centre];
  const why = community.words
    ? `What marks them out: <b>${esc(community.words)}</b>, which most of them have and
      few games outside do.`
    : 'No one thing marks them out; they are alike in many small ways.';
  const places = (community.parts || []).length;
  return `<h3 style="margin-top:0">${esc(community.name)}</h3>
    <div class="small">${community.size} games, each among the nearest of others here
      rather than of the rest of the library. ${why}${centre
        ? ` The one most like the rest of them is <b>${esc(centre.name)}</b>.` : ''}</div>
    ${places > 1 ? `<div class="muted small" style="margin-top:8px">Drawn in ${places}
      places: the map keeps each game beside the games most like it, and parts of this
      community share little but what names it.</div>` : ''}
    ${community.size >= MAP_COLOURED ? `<div class="row" style="margin-top:10px">
      <button data-fold-group="${esc(community.id)}">fold it into one dot</button></div>`
      : ''}
    <div class="links">${community.members.map((hash, at) => `
      <button class="linkRow" data-node-row="${esc(hash)}">
        <b>${esc(community.names[at])}</b></button>`).join('')}</div>`;
}

// The design steps that make another game out of this one, once the server has said
// which they are. A step is one of the invention loop's own edits and the words are its
// own, so nothing here decides what an edit is or which of them apply -- the press asks,
// and the panel prints what came back. The list is not asked for until somebody presses
// *and now try*, because working it out means drawing every step against this game.
function mapTryHtml(node) {
  const held = MAP_TRY[node.hash];
  if (!held) return '';
  if (held.state === 'asking') {
    return `<div class="muted small" style="margin-top:10px">Looking for the small changes
      to the rules that apply here&hellip;</div>`;
  }
  if (held.state === 'failed') {
    return `<div class="muted small" style="margin-top:10px">${esc(held.why)}</div>`;
  }
  if (!held.steps.length) {
    return `<div class="muted small" style="margin-top:10px">No small change to the rules
      makes a different game from this one.</div>`;
  }
  return `<div class="muted small" style="margin-top:10px">${held.steps.length === 1
      ? 'One small change to the rules makes' : held.steps.length + ` small changes to the
      rules each make`} a different game from this one. Pressing one opens it in the
      editor, which keeps nothing until you save it.${held.making
        ? ' <b>Making it…</b>' : ''}</div>
    <div class="links">${held.steps.map(step => `
      <button class="linkRow" data-step="${esc(step.operator)}"
        data-seed="${esc(String(step.seed))}" data-from="${esc(node.hash)}">
        <div class="small">${esc(step.words)}</div>
      </button>`).join('')}</div>`;
}

// One line's gap, read from the end a reader is standing at. The payload signs it from
// the line's own source, so seen from the other end the two numbers change places, each
// with its run count, and the difference changes sign. Nothing else about it moves, and
// the words stay `gapLine`'s --
// the same sentence the game's own page prints about the same pair.
function mapGapFrom(line, hash) {
  const gap = line.gap;
  if (!gap || line.source === hash) return gap || null;
  return {
    ...gap,
    index: gap.otherIndex,
    otherIndex: gap.index,
    seeds: gap.otherSeeds,
    otherSeeds: gap.seeds,
    gap: -gap.gap,
    moved: (gap.moved || []).map(part => ({
      ...part, index: part.otherIndex, otherIndex: part.index,
    })),
  };
}

// ------------------------------------------------------------- opening the editor on it
//
// The server hands back a draft, the editor takes it, and nothing is written down
// anywhere until somebody presses save in there. The draft carries its own description,
// which is the server's sentence about what was changed -- the words of the design step.

function mapAskSteps(hash) {
  if (MAP_TRY[hash] && MAP_TRY[hash].state !== 'failed') return;
  MAP_TRY[hash] = { state: 'asking' };
  paintMapPanel();
  api('/api/design/' + hash + '?steps=1').then(answer => {
    MAP_TRY[hash] = { state: 'ready', steps: answer.steps || [] };
    paintMapPanel();
  }).catch(error => {
    MAP_TRY[hash] = { state: 'failed', why: errorWords(error) };
    paintMapPanel();
  });
}

function mapTakeStep(hash, step, seed) {
  // Said in the panel rather than over the whole page: the map is still what a reader is
  // looking at, and a step that cannot be made has to answer beside the step they pressed.
  if (MAP_TRY[hash]) MAP_TRY[hash].making = step;
  paintMapPanel();
  api('/api/design/' + hash + '?step=' + encodeURIComponent(step) + '&seed=' + seed)
    .then(answer => mapOpenDraft(answer, answer.remix))
    .catch(error => {
      MAP_TRY[hash] = { state: 'failed', why: errorWords(error) };
      paintMapPanel();
    });
}

// The editor holds its draft in the page's own state, so a remix puts one there and goes
// to it. `showDesigner` with no game in the route keeps the draft it finds, which is what
// makes this the handover rather than a second way of opening the editor.
function mapOpenDraft(answer, made) {
  DRAFT = made.spec;
  DRAFT_FROM = answer.hash;
  // What the game it came from *is*, and what its own measurement cost. The editor says
  // that a changed rule leaves that number behind, and this is the same payload *make
  // one like this* hands it, because it is the same route.
  DRAFT_PARENT = answer;
  DRAFT_PIECE = 0;
  // Everything the editor was holding about some earlier draft. A check, a rung or a
  // saved message left standing would be a sentence about another game.
  DRAFT_CHECK = null;
  DRAFT_PREVIEW = null;
  DRAFT_MAP = null;
  DRAFT_HASH = '';
  DRAFT_STANDING = null;
  DRAFT_SCREEN = null;
  DRAFT_SAVED = '';
  // A remix is where undo starts. There is nothing behind it to go back to, and the
  // editor's own first snapshot is this draft.
  DRAFT_UNDO = [];
  DRAFT_REDO = [];
  DRAFT_SNAP = '';
  keepDraft();
  go('#/design');
}

// What the two scores make of this line: *what did the difference do*. The sentence is
// `gapLine`'s, the game page's own, so one pair cannot read `clear` in one place and `no
// clear change` in the other; the line above it names which game the sentence is about,
// because here there are two of them.
//
// Where there is no gap the panel says which of the two cannot be read, or that the two
// were scored under different rules -- "measured and alike" and "not measured" are
// different news, and so is "measured under something else".
function mapGapHtml(line, from, to) {
  if (line.gap) {
    return `<div class="small" style="margin-top:10px"><b>${esc(to.name)}</b> against
      <b>${esc(from.name)}</b></div>${gapLine(line.gap)}${mapMovedHtml(line.gap)}`;
  }
  const missing = [from, to].filter(end => !end.score).map(end => end.name);
  if (!missing.length) {
    return `<div class="muted small" style="margin-top:10px">These two were scored under
      different versions of the scoring rules, which never share a scale, so there is no
      difference to take.</div>`;
  }
  return `<div class="muted small" style="margin-top:10px">No score to compare &mdash;
    ${esc(missing.join(' and '))} ${missing.length === 1 ? 'has' : 'have'} ${ACCOUNTS
      ? 'no current score' : 'no measurement this checkout can read'}.</div>`;
}

// Which parts of the score moved, under the number that says how far the whole of it did.
// A game's index is a dozen measurements combined, so *this rule cost 0.09* is a headline
// with nothing under it; these are the three that moved the index most, each with its own
// number at both ends. Which three, and by how much, is the payload's -- the page neither
// ranks them nor works out a difference.
function mapMovedHtml(gap) {
  const moved = (gap && gap.moved) || [];
  if (!moved.length) return '';
  return `<div class="muted small" style="margin-top:10px">What changed most, of the
      eleven things measured</div>
    <div class="links">${moved.map(part => `
      <div class="linkRow">
        <div class="row" style="gap:8px; align-items:baseline">
          <b>${esc(mapComponentWords(part.key))}</b>
          <span class="muted small">${esc(fmt(part.otherIndex))} against ${
            esc(fmt(part.index))}</span>
        </div>
      </div>`).join('')}</div>`;
}

// What a visitor may fold into one dot and see outlined: the communities of three or more,
// which are the ones with a colour of their own. A pair is already one line between two
// dots, and folding it saves nothing. The communities are a partition, so a game folds into
// one place or none, and each carries the place its folded dot opens at (`x`, `y`).
function mapGroups() {
  return mapCommunities().filter(one => one.size >= MAP_COLOURED);
}

function mapGroupAt(index) {
  return mapGroups()[index] || null;
}

// A measurement's name in words. `metrics/glossary.py` decides it and the payload carries
// it, for the same reason the panel never prints a kind's schema name.
function mapComponentWords(key) {
  return (MAP_DATA.components || {})[key] || key;
}

// Where a design run made the child (M8): the run's name is the maintainer's, so the site
// says the computer invented it and after how many rounds of changes.
function mapInventedWords(arrow) {
  const known = arrow.generation !== null && arrow.generation !== undefined;
  if (ACCOUNTS) {
    return `The computer invented it${known ? `, after ${arrow.generation} ${
      Number(arrow.generation) === 1 ? 'round' : 'rounds'} of changes` : ''}.`;
  }
  return `It came out of the design run ${esc(arrow.run)}${
    known ? ', generation ' + esc(arrow.generation) : ''}.`;
}

function mapArrowPanel(arrow) {
  const from = arrow && MAP_AT[arrow.source], to = arrow && MAP_AT[arrow.target];
  if (!from || !to) return mapNothingPanel();
  return `<h3 style="margin-top:0">Made from</h3>
    <div class="small"><b>${esc(to.name)}</b> was made as a new version of
      <b>${esc(from.name)}</b>, and the two are ${arrow.units === 1
        ? 'one rule' : arrow.units + ' rules'} apart. An arrow is drawn however far apart
      they are, because what came from what is a fact no comparison of two rule sets can
      find.${arrow.run ? ` ${mapInventedWords(arrow)}` : ''}</div>
    <div class="row" style="margin-top:10px">
      <button data-open="${esc(arrow.source)}">open ${esc(from.name)}</button>
      <button data-open="${esc(arrow.target)}">open ${esc(to.name)}</button>
    </div>`;
}

// A community folded into one dot, which is the explorer's doing: the panel is here
// because every panel is, so one library is described in one place.
function mapGroupPanel(group) {
  if (!group) return mapNothingPanel();
  const why = group.words
    ? `What marks them out: <b>${esc(group.words)}</b>.`
    : 'No one thing marks them out; they are alike in many small ways.';
  return `<h3 style="margin-top:0">${esc(group.name)}</h3>
    <div class="small">${group.size} games of one community, folded into one dot. ${why}
      Open it to put every one of them back where it was; the lines this dot carries are
      the lines its games have to the rest of the library.</div>
    <div class="row" style="margin-top:10px">
      <button data-open-group="${esc(group.id)}">open this community</button></div>
    <div class="links">${group.names.map((name, at) => `
      <button class="linkRow" data-node-row="${esc(group.members[at])}">
        <b>${esc(name)}</b></button>`).join('')}</div>`;
}

// A kind of rule in words, from the payload's `kinds`, which `graphing.KIND_WORDS` decides.
function mapKindWords(kind) {
  const found = MAP_DATA.kinds.find(entry => entry.kind === kind);
  return (found && found.words) || kind;
}
