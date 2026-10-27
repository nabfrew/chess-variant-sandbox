// The board's motion, one copy for every page that draws a game moving (LP4,
// `docs/archive/plan-landing.md` §2.2 and §3.3): the play board (`board.js`, `simultaneous.js`) and
// the landing's played stories (`landing.js`). A piece's travel, a capture's blow, a stone
// dropping in, the portal journey and the blast are drawn here and nowhere else, so a change
// to how a blast looks changes it on both pages (the user, 2026-10-24: *"The landing page
// should use the animations for portal and explosions that the actual gameplay uses"*; the
// blast is the landing mock-up's, which the play board took in its place).
//
// What moved where, through which portal, and what a blast cleared are the caller's to say,
// read off the engine (rule 3): every function here takes squares, positions on the drawing
// and times, and works out nothing about the game. Geometry is the drawing's (`squareXY`,
// `viz/wrapped.js`), so the same calls serve an SVG drawn by the play board and one drawn by
// the still board.
//
// Everything is the Web Animations API (`element.animate`) and CSSOM writes, which the site's
// `style-src 'self'` allows: no style attribute and no style element.
//
// A piece is drawn by the page that owns the drawing: `motionDrawer(svg, draw)` names how a
// ghost of a piece is drawn on that svg (`draw(piece, x, y, cell)` returns its markup); an svg
// with none is the play board's, drawn by its `drawPiece`.

// Every duration the board uses is stated here, in milliseconds, and nowhere else. It is
// deliberately unhurried -- a move is something a person has to *read*, and the pace this
// replaces was fast enough that a capture was finished before the capturing piece had
// arrived.
const PACE = {
  travelMin: 225,        // a move of no distance at all
  travelPerUnit: 21,     // ... and this much more per viewBox unit crossed
  travelMax: 630,        // ... but never longer than this, however wide the board
  impact: 0.8,           // the share of the travel after which the victim is struck
  strike: 390,           // a captured piece being knocked off the board
  burst: 330,            // the ring that marks where it was taken
  drop: 330,             // a piece arriving from a hand
  claim: 420,            // a claimed square's paint growing from its centre (LP5)
  // A blast (LP4, the landing mock-up's): the fireball over the area, swelling and gone in
  // this long; each piece it took burns and is thrown in this long, starting this much
  // after the blast goes off.
  blastFire: 750,
  blastBurn: 690,
  blastLag: 60,
  resist: 500,           // the shield shimmer on a piece that survived the blast
  rotate: 350,           // a piece turning to a new facing
  // A sweep (schema 13, `sweepCut`): each piece it passes is cut and its halves thrown
  // apart in this long, the cut's stroke flashing and its sparks flying in these long, and
  // the blade left along the line fades in this long once the sweeper lands.
  sweepSplit: 720,
  sweepSlash: 280,
  sweepSparks: 420,
  sweepFade: 360,
  // A void move in a round (schema 26, Q918c): the piece steps toward the square it went
  // for and back, in this long, reaching this share of the way.
  voidStep: 460,
  voidReach: 0.42,
  // The gap between a ply's two beats -- the move somebody made, and the wild-piece
  // phase the rules run after it. Not a delay put in to hide a race: the phase happens
  // after the move, so it is *drawn* after the move, and this is only the pause
  // between them. Shorter than the shortest travel (225ms), so a machine game does not
  // crawl; long enough that the piece that moved has landed and come to rest before
  // anything else on the board starts moving, which is the whole of what was missing.
  beat: 200,
  // A move through a portal (schema 19, Q836, `docs/archive/plan-portals.md` *Travel through a
  // portal, animated*): one journey in three movements, laid out by `portalJourney`. The
  // slide to the end it goes in by, quicker than a travel because the portal pulls ...
  portalApproachMin: 170,
  portalApproachPerUnit: 7,
  portalApproachMax: 320,
  // ... the share of that slide after which the dive starts, so the piece is already
  // winding down into the rim as it arrives rather than stopping and then shrinking ...
  portalOverlap: 0.72,
  // ... the shrink and spin down into it ...
  portalDive: 250,
  // ... the instant it is nowhere, between the two ends ...
  portalVoid: 40,
  // ... and the grow and spin up out of the other end, the one step on, and the settle.
  portalEmerge: 400,
  // The share of the emergence after which a piece standing where it lands is struck:
  // the moment the piece, full size again, reaches that square.
  portalImpact: 0.6,
  // With reduced motion: no slide, no spin, no scale; the piece fades from where it was
  // and into where it lands, the two overlapping, in this long all told.
  portalStill: 300,
  // A move through a fold (schema 24, `docs/archive/plan-portals-2.md` §5.3R), laid out by
  // `foldJourney`: the slide onto the fold is the portal's approach; then the sink into the
  // pool (no spin: it goes through a surface, not down a drain) ...
  foldSink: 240,
  // ... the instant under the surface ...
  foldUnder: 60,
  // ... and the rise out of the far end, where the move ends.
  foldRise: 260,
  // A ring of ripple in the hue, where it sinks and where it rises.
  foldRipple: 520,
};
const SVGNS = 'http://www.w3.org/2000/svg';
// How a ghost of a piece is drawn on each drawing the motion plays over (see the top).
const MOTION_DRAWERS = new WeakMap();
function motionDrawer(svg, draw) {
  MOTION_DRAWERS.set(svg, draw);
}
// ... and on an svg with none: the play board's `drawPiece`, which `board.js` hands over as it
// loads, so a page that loads this file without the play board (the user page, UP5) names
// nothing it lacks.
let MOTION_PIECE = null;

// A blast (LP4; the landing mock-up's, which replaced the play board's ring per victim): one
// fireball over the whole area the rule's table names, swelling and fading, and each piece
// the blast took burning and thrown outward from its centre (`burn`), while the pieces it
// spared stand (`resist`). `area` is the squares the caller read off the compiled blast
// table, the centre among them; `at` is when the blast goes off. Returns when it is over.
function blastFire(svg, area, cell, view, at) {
  const boxes = area.map(square => squareXY(square, cell, view));
  if (!boxes.length) return at;
  const x0 = Math.min(...boxes.map(b => b[0])), y0 = Math.min(...boxes.map(b => b[1]));
  const x1 = Math.max(...boxes.map(b => b[0])) + cell, y1 = Math.max(...boxes.map(b => b[1])) + cell;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, rx = (x1 - x0) / 2, ry = (y1 - y0) / 2;
  const fire = document.createElementNS(SVGNS, 'g');
  fire.setAttribute('data-journey', 'blast');
  fire.setAttribute('pointer-events', 'none');
  // Flare outside, fire, then the spark at its heart: the mock-up's radial gradient as three
  // rings of the effect tokens, so it needs no gradient definition of its own.
  fire.innerHTML = [['var(--fx-flare)', 1, 0.5], ['var(--fx-fire)', 0.62, 0.8],
    ['var(--fx-spark)', 0.26, 1]].map(([ink, k, opacity]) =>
    `<ellipse cx="${cx}" cy="${cy}" rx="${rx * k}" ry="${ry * k}" fill="${ink}" opacity="${opacity}"/>`)
    .join('');
  fire.style.opacity = '0';
  fire.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(fire);
  const run = fire.animate([
    { opacity: 0, transform: 'scale(0.15)' },
    { opacity: 1, transform: 'scale(0.7)', offset: 0.25 },
    { opacity: 0, transform: 'scale(1.15)' },
  ], { duration: PACE.blastFire, delay: at, easing: 'ease-out', fill: 'forwards' });
  run.onfinish = () => fire.remove();
  return at + PACE.blastFire;
}

// A piece the blast took: it flares, swells, and is thrown outward from the blast's centre
// (`from`, the centre's corner on the drawing, or its own square for a blast with none
// named), shrinking and turning as it goes. Returns when it is gone.
function burn(svg, piece, x, y, cell, blow, from) {
  const ghost = ghostAt(svg, piece, x, y, cell);
  ghost.setAttribute('data-journey', 'burn');
  const [ox, oy] = from || [x, y];
  const dx = (x - ox) * 0.6, dy = (y - oy) * 0.6 - cell * 0.3;
  const spin = dx > 0 ? 60 : dx < 0 ? -60 : 0;
  const start = blow.at + PACE.blastLag;
  const run = ghost.animate([
    { opacity: 1, transform: 'translate(0px, 0px) scale(1) rotate(0deg)', filter: 'brightness(1)' },
    { opacity: 1, transform: 'translate(0px, 0px) scale(1.15) rotate(0deg)', filter: 'brightness(1.8)',
      offset: 0.3 },
    { opacity: 0, transform: `translate(${dx}px, ${dy}px) scale(0.3) rotate(${spin}deg)`,
      filter: 'brightness(2)' },
  ], { duration: PACE.blastBurn, delay: start, easing: 'ease-in', fill: 'forwards' });
  run.onfinish = () => ghost.remove();
  return start + PACE.blastBurn;
}

// A still board's last move played once (UP5, `docs/plan-user-page.md` §5, the user page's
// waiting games): the piece standing on `last.to` in a still drawn with `preview.motion`
// slides there from `last.from` (`travel`), or drops in from a hand where the move had no
// origin (`stoneDrop`). `preview` is the payload the still was drawn from, so the geometry is
// the drawing's own, as the landing's stories ask it. Returns how long it takes, 0 where the
// drawing has no piece there.
function stillMove(svg, preview, last) {
  const piece = svg && svg.querySelector(`[data-piece="${last.to}"]`);
  if (!piece) return 0;
  const W = preview.width, H = preview.height, cell = 100 / Math.max(W, H);
  const view = flatView(W, H, FLIPPED, preview.lattice, boardCut(preview.absent, preview.wrap));
  const [tx, ty] = squareXY(last.to, cell, view);
  if (last.from === null || last.from === undefined) return stoneDrop(piece, tx, ty, cell);
  const [fx, fy] = squareXY(last.from, cell, view);
  return travel(piece, fx - tx, fy - ty, false);
}

// A piece arriving from a hand: it drops in from above and settles. `delay` holds it unseen
// until its moment. `x`, `y` is its square's corner on the drawing; `most` caps how long it
// takes, as `travel`'s does. Returns when it has settled.
function stoneDrop(node, x, y, cell, delay = 0, most = Infinity) {
  const duration = Math.min(most, PACE.drop);
  node.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  const run = node.animate([
    { opacity: 0, transform: `translate(0px, ${-cell * 0.55}px) scale(1.5)` },
    { opacity: 1, transform: `translate(0px, ${cell * 0.04}px) scale(0.96)`, offset: 0.75 },
    { opacity: 1, transform: 'translate(0px, 0px) scale(1)' },
  ], { duration: duration, delay: delay, easing: 'ease-in', fill: delay ? 'backwards' : 'none' });
  run.onfinish = () => { node.style.transformOrigin = ''; };
  return delay + duration;
}

// A claim (LP5, §2.2): the square's paint grows from its centre in the claimer's colour,
// over whatever paint was there, starting `at`. `paint` is the square's paint already drawn
// in the new colour; `x`, `y` its square's corner. `most` caps how long it takes, as
// `travel`'s does. Returns when it is fully painted.
function claimPaint(paint, x, y, cell, at = 0, most = Infinity) {
  const duration = Math.min(most, PACE.claim);
  paint.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  const run = paint.animate([
    { transform: 'scale(0)', opacity: 0.6 },
    { transform: 'scale(1)', opacity: 1 },
  ], { duration: duration, delay: at, easing: 'ease-out', fill: 'backwards' });
  run.onfinish = () => { paint.style.transformOrigin = ''; };
  return at + duration;
}

// Returns how long the move takes, which is what a capture on the destination square
// has to wait for.
//
// `delay` is which beat of the ply this is. It is `backwards`-filled on purpose: a piece
// whose beat has not started yet is drawn where it *was*, not where it ends up, which is
// what lets a wild piece stand still on the square it came from while the player's move
// plays out in front of it.
//
// `most`, where given, caps the travel: a game played at a story's pace (the landing's)
// moves each piece within its turn.
function travel(node, dx, dy, arc, delay = 0, most = Infinity) {
  const distance = Math.hypot(dx, dy);
  const duration = Math.min(most, PACE.travelMax,
                            PACE.travelMin + distance * PACE.travelPerUnit);
  const frames = [{ transform: `translate(${dx}px, ${dy}px)` }];
  if (arc) {
    // A lift and a slight swell, so a leap reads as going over rather than through.
    const lift = Math.min(distance * 0.45, 6);
    frames.push({ transform: `translate(${dx / 2}px, ${dy / 2 - lift}px) scale(1.12)`, offset: 0.5 });
  }
  frames.push({ transform: 'translate(0px, 0px)' });
  node.animate(frames, { duration, delay, fill: delay ? 'backwards' : 'none',
                         easing: arc ? 'ease-in-out' : 'cubic-bezier(.3,.7,.3,1)' });
  return duration;
}

// A capture, in two parts, because either one alone is easy to miss: the victim is
// knocked off the square it was standing on, and a ring is struck where it stood. The
// ring is the part that reads at a glance on a crowded board, and it is the reason this
// is not simply a fade -- a piece disappearing is what *any* of these three facts looks
// like, and a capture is the one worth telling apart.
//
// `blow` says when the piece that killed this one arrives and which way it came from, so
// both halves are timed to the impact rather than to the redraw. Returns when the last of
// it finishes.
function capture(svg, piece, x, y, cell, blow) {
  return strike(svg, ghostAt(svg, piece, x, y, cell), x, y, cell, blow);
}

// A piece drawn on the board that the *position* no longer has: the victim of a capture,
// or a piece that moved onto a square something else has since taken it on. It stands
// there, at full opacity, until whatever happens to it happens -- an animation that has
// not started leaves its element showing its own style.
function ghostAt(svg, piece, x, y, cell) {
  const ghost = document.createElementNS(SVGNS, 'g');
  ghost.innerHTML = (MOTION_DRAWERS.get(svg) || MOTION_PIECE)(piece, x, y, cell);
  ghost.setAttribute('pointer-events', 'none');
  ghost.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  svg.appendChild(ghost);
  return ghost;
}

// The taking itself, done to a ghost that is already on the board. Split out from
// `capture` because a piece can travel *first* and be taken where it landed -- the phase
// eating what the mover just moved into it -- and that has to be one piece the eye can
// follow rather than one that vanishes and another that appears.
function strike(svg, ghost, x, y, cell, blow) {
  const cx = x + cell / 2, cy = y + cell / 2;
  const reach = Math.hypot(blow.dx, blow.dy) || 1;
  const kx = blow.dx / reach * cell * 0.24, ky = blow.dy / reach * cell * 0.24;

  const fade = ghost.animate([
    { opacity: 0.95, transform: 'translate(0px, 0px) scale(1)' },
    { opacity: 0.8, transform: `translate(${kx * 0.55}px, ${ky * 0.55}px) scale(1.08)`,
      offset: 0.3 },
    { opacity: 0, transform: `translate(${kx}px, ${ky}px) scale(0.5)` },
  ], { duration: PACE.strike, delay: blow.at, easing: 'ease-in', fill: 'forwards' });
  fade.onfinish = () => ghost.remove();

  const ring = document.createElementNS(SVGNS, 'circle');
  ring.setAttribute('cx', cx);
  ring.setAttribute('cy', cy);
  ring.setAttribute('r', cell * 0.4);
  ring.setAttribute('fill', 'none');
  ring.setAttribute('stroke', 'var(--bad)');
  ring.setAttribute('stroke-width', cell * 0.07);
  ring.setAttribute('pointer-events', 'none');
  // Invisible until the blow lands: an animation that has not started yet leaves the
  // element showing its own style, and a ring sitting on the victim beforehand would
  // announce the capture before it happened.
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(ring);
  const flash = ring.animate([
    { opacity: 0.9, transform: 'scale(0.45)' },
    { opacity: 0, transform: 'scale(1.3)' },
  ], { duration: PACE.burst, delay: blow.at, easing: 'ease-out', fill: 'forwards' });
  flash.onfinish = () => ring.remove();

  return blow.at + Math.max(PACE.strike, PACE.burst);
}

// A sweep (schema 13): not a blow on one square but an edge drawn down a line. A blade
// streaks out behind the sweeper as it travels, and each piece it passes over is cut along
// the line at the moment the sweeper's edge reaches it: a flash of the cut, sparks off
// the edge, and the two halves thrown apart to either side, turning away and gone. Told apart from a capture's
// knock and a blast's burn on purpose, because a sweep takes pieces nothing landed on.
//
// `mover` is the sweeper's node, which the blade is laid under; `from` and `to` its two
// squares' corners on the drawing; `victims` is `[piece, x, y]` for each square the engine
// says it cleared (`swept`, `viz.wrapped.sweep_transit`), in the order it passed them;
// `flight` is how long `travel` gave the sweeper, and `delay` when it set off. Returns when
// the last half is gone.
let SWEEP_CUTS = 0;
function sweepCut(svg, mover, from, to, victims, cell, flight, delay = 0) {
  const draw = MOTION_DRAWERS.get(svg) || MOTION_PIECE;
  const h = cell / 2;
  const [ax, ay] = [from[0] + h, from[1] + h], [bx, by] = [to[0] + h, to[1] + h];
  const length = Math.hypot(bx - ax, by - ay) || 1;
  const ux = (bx - ax) / length, uy = (by - ay) / length, nx = -uy, ny = ux;
  // Where the sweeper is at a share of its flight: `travel`'s easing for a straight move,
  // cubic-bezier(.3, .7, .3, 1), solved for the moment it has covered `share` of the line.
  const when = share => {
    const bez = (u, p, q) => 3 * (1 - u) * (1 - u) * u * p + 3 * (1 - u) * u * u * q + u * u * u;
    let lo = 0, hi = 1;
    for (let i = 0; i < 20; i++) {
      const mid = (lo + hi) / 2;
      if (bez(mid, 0.7, 1) < share) lo = mid; else hi = mid;
    }
    return delay + flight * bez((lo + hi) / 2, 0.3, 0.3);
  };

  // The blade: a broad faint wake and a bright edge inside it, drawn out from the origin as
  // the sweeper goes, then thinning away once it has landed.
  const blade = document.createElementNS(SVGNS, 'g');
  blade.setAttribute('data-journey', 'sweep');
  blade.setAttribute('pointer-events', 'none');
  blade.innerHTML = [['var(--fx-edge)', 0.34, 0.4], ['var(--fx-blade)', 0.08, 0.95]]
    .map(([ink, width, opacity]) =>
      `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}" stroke="${ink}" `
      + `stroke-width="${cell * width}" stroke-linecap="round" opacity="${opacity}" `
      + 'pathLength="1" stroke-dasharray="1 1" stroke-dashoffset="1"/>').join('');
  if (mover && mover.parentNode === svg) svg.insertBefore(blade, mover);
  else svg.appendChild(blade);
  for (const line of blade.querySelectorAll('line')) {
    line.animate([{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }],
      { duration: flight, delay, easing: 'cubic-bezier(.3,.7,.3,1)', fill: 'forwards' });
  }
  const fade = blade.animate([{ opacity: 1 }, { opacity: 0 }],
    { duration: PACE.sweepFade, delay: delay + flight, easing: 'ease-in', fill: 'forwards' });
  fade.onfinish = () => blade.remove();
  let done = delay + flight + PACE.sweepFade;

  for (const [piece, x, y] of victims) {
    const cx = x + h, cy = y + h;
    const at = when(Math.min(1, Math.max(0, ((cx - ax) * ux + (cy - ay) * uy) / length)));
    // Two copies of the piece, each clipped to one side of the line through its centre, so
    // that standing together they are the piece and nothing has happened yet.
    const id = `sweep-cut-${++SWEEP_CUTS}`;
    [1, -1].forEach((side, k) => {
      const corner = (u, n) => `${cx + ux * u * cell + nx * n * cell},${cy + uy * u * cell + ny * n * cell}`;
      const half = document.createElementNS(SVGNS, 'g');
      half.setAttribute('data-journey', 'sweep-half');
      half.setAttribute('pointer-events', 'none');
      half.innerHTML = `<clipPath id="${id}-${k}"><polygon points="${corner(-1, 0)} ${corner(1, 0)} `
        + `${corner(1, side)} ${corner(-1, side)}"/></clipPath>`
        + `<g clip-path="url(#${id}-${k})">${draw(piece, x, y, cell)}</g>`;
      half.style.transformOrigin = `${cx}px ${cy}px`;
      svg.appendChild(half);
      const ox = nx * side * cell, oy = ny * side * cell;
      // A hitch as the edge goes through, then thrown apart, carried a little the way the
      // sweeper went, each half turning away from the line.
      const away = (k, turn) => `translate(${ox * k + ux * cell * k * 0.4}px, `
        + `${oy * k + uy * cell * k * 0.4}px) rotate(${side * turn}deg)`;
      const run = half.animate([
        { opacity: 1, transform: away(0, 0), filter: 'brightness(1)' },
        { opacity: 1, transform: away(0.08, 0), filter: 'brightness(1.8)', offset: 0.14 },
        { opacity: 0.9, transform: away(0.45, 20), filter: 'brightness(1.2)', offset: 0.55 },
        { opacity: 0, transform: away(0.8, 40), filter: 'brightness(1)' },
      ], { duration: PACE.sweepSplit, delay: at, easing: 'cubic-bezier(.2,.6,.4,1)',
           fill: 'forwards' });
      run.onfinish = () => half.remove();
    });
    // The cut itself: a bright stroke along the line across the piece, there and gone.
    const slash = document.createElementNS(SVGNS, 'line');
    slash.setAttribute('x1', cx - ux * cell * 0.48);
    slash.setAttribute('y1', cy - uy * cell * 0.48);
    slash.setAttribute('x2', cx + ux * cell * 0.48);
    slash.setAttribute('y2', cy + uy * cell * 0.48);
    slash.setAttribute('stroke', 'var(--fx-blade)');
    slash.setAttribute('stroke-width', cell * 0.06);
    slash.setAttribute('stroke-linecap', 'round');
    slash.setAttribute('pointer-events', 'none');
    slash.setAttribute('pathLength', '1');
    slash.setAttribute('stroke-dasharray', '1 1');
    slash.style.opacity = '0';
    svg.appendChild(slash);
    // Drawn along the line the way the sweeper is going, held a moment, and gone.
    const flash = slash.animate([
      { opacity: 1, strokeDashoffset: 1 },
      { opacity: 1, strokeDashoffset: 0, offset: 0.35 },
      { opacity: 0, strokeDashoffset: 0 },
    ], { duration: PACE.sweepSlash, delay: at, easing: 'ease-out', fill: 'forwards' });
    flash.onfinish = () => slash.remove();
    // Sparks off the edge, thrown on ahead along the line and fanning to either side.
    for (let i = 0; i < 5; i++) {
      const fan = (i - 2) * 0.45, reach = cell * (0.45 + 0.12 * (i % 3));
      const spark = document.createElementNS(SVGNS, 'circle');
      spark.setAttribute('cx', cx);
      spark.setAttribute('cy', cy);
      spark.setAttribute('r', cell * 0.035);
      spark.setAttribute('fill', i % 2 ? 'var(--fx-edge)' : 'var(--fx-blade)');
      spark.setAttribute('pointer-events', 'none');
      spark.style.opacity = '0';
      svg.appendChild(spark);
      const sx = (ux + nx * fan) * reach, sy = (uy + ny * fan) * reach;
      const fly = spark.animate([
        { opacity: 1, transform: 'translate(0px, 0px)' },
        { opacity: 0, transform: `translate(${sx}px, ${sy}px)` },
      ], { duration: PACE.sweepSparks, delay: at, easing: 'cubic-bezier(.1,.8,.3,1)',
           fill: 'forwards' });
      fly.onfinish = () => spark.remove();
    }
    done = Math.max(done, at + Math.max(PACE.sweepSplit, PACE.sweepSparks));
  }
  return done;
}
// A resist shimmer: the piece was inside the blast radius but is immune to the
// explosion. A translucent shield briefly pulses around it, so the player sees
// "that piece was in danger and survived" rather than "nothing happened there".
function resist(svg, x, y, cell, delay) {
  const cx = x + cell / 2, cy = y + cell / 2;

  // A rounded-rect shield outline that pulses once and fades.
  const shield = document.createElementNS(SVGNS, 'rect');
  const pad = cell * 0.1;
  shield.setAttribute('x', x + pad);
  shield.setAttribute('y', y + pad);
  shield.setAttribute('width', cell - pad * 2);
  shield.setAttribute('height', cell - pad * 2);
  shield.setAttribute('rx', cell * 0.15);
  shield.setAttribute('ry', cell * 0.15);
  shield.setAttribute('fill', 'none');
  shield.setAttribute('stroke', 'var(--fx-shield)');
  shield.setAttribute('stroke-width', cell * 0.07);
  shield.setAttribute('pointer-events', 'none');
  shield.style.opacity = '0';
  shield.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(shield);
  const pulse = shield.animate([
    { opacity: 0, transform: 'scale(1.15)' },
    { opacity: 0.75, transform: 'scale(1)', offset: 0.3 },
    { opacity: 0.6, transform: 'scale(1.05)', offset: 0.6 },
    { opacity: 0, transform: 'scale(1.1)' },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  pulse.onfinish = () => shield.remove();

  // A faint filled overlay that flashes to reinforce the "shielded" read.
  const glow = document.createElementNS(SVGNS, 'rect');
  glow.setAttribute('x', x + pad);
  glow.setAttribute('y', y + pad);
  glow.setAttribute('width', cell - pad * 2);
  glow.setAttribute('height', cell - pad * 2);
  glow.setAttribute('rx', cell * 0.15);
  glow.setAttribute('ry', cell * 0.15);
  glow.setAttribute('fill', 'var(--fx-shield)');
  glow.setAttribute('pointer-events', 'none');
  glow.style.opacity = '0';
  glow.style.transformOrigin = `${cx}px ${cy}px`;
  svg.appendChild(glow);
  const flash = glow.animate([
    { opacity: 0 },
    { opacity: 0.2, offset: 0.3 },
    { opacity: 0 },
  ], { duration: PACE.resist, delay, easing: 'ease-in-out', fill: 'forwards' });
  flash.onfinish = () => glow.remove();

  return delay + PACE.resist;
}
// A trip through a portal (Q836, `docs/archive/plan-portals.md` *Travel through a portal,
// animated*), as times in milliseconds from the start of the move: the slide to the end
// it goes in by (`approach`), the shrink and spin down into it (`dive`), which starts
// before the slide is quite over so the two read as one movement, the instant it is
// nowhere, the grow and spin up out of the other end and the step on to where it lands
// (`emerge`), the blow a piece standing there is struck with (`strike`), and when it is
// at rest (`landed`). `distance` is the slide's length in viewBox units. Pure, so the
// order of the steps is something a test can hold.
function portalJourney(distance) {
  const approach = Math.min(PACE.portalApproachMax,
                            PACE.portalApproachMin + distance * PACE.portalApproachPerUnit);
  const dive = { at: approach * PACE.portalOverlap, for: PACE.portalDive };
  const emerge = { at: dive.at + dive.for + PACE.portalVoid, for: PACE.portalEmerge };
  return {
    approach: { at: 0, for: approach }, dive: dive, emerge: emerge,
    strike: emerge.at + emerge.for * PACE.portalImpact,
    landed: emerge.at + emerge.for,
  };
}

// The slide, for the piece going in and, on a slide across the seam, for the copy of it
// leaving by the near edge: both the same motion, one repeat apart. Eased in, with a hair
// of give at the start, so it gathers speed into the portal as though pulled, rather
// than coasting to a stop beside it; a leap lifts over what it passes, as a travel does.
function portalSlide(el, dx, dy, arc, duration) {
  const frames = [{ transform: `translate(${dx}px, ${dy}px)` }];
  if (arc) {
    const lift = Math.min(Math.hypot(dx, dy) * 0.45, 6);
    frames.push({ transform: `translate(${dx / 2}px, ${dy / 2 - lift}px) scale(1.1)`,
                  offset: 0.5 });
  }
  frames.push({ transform: 'translate(0px, 0px)' });
  el.animate(frames, { duration: duration, fill: 'forwards',
                       easing: arc ? 'cubic-bezier(.45,0,.7,.6)' : 'cubic-bezier(.5,-.12,.75,.5)' });
}
// A dashed ring in its pair's colour, edged so it reads on both woods, turning the way the
// piece turns: drawn in towards the centre as the piece goes down (`inward`), thrown out
// from it as the piece comes up. Laid under `under` when there is one, so the piece is
// seen inside it. Gone when it is done; invisible before it starts.
//
// Its look is a class and two custom properties, never a colour of its own: the ring is
// `.portal-swirl` (with `data-swirl` in or out and the end's `data-portal`, `data-pair`
// and `data-end`), stroked in `--swirl-ink` and edged in `--swirl-edge`, each falling back
// to `portalLook`'s colour and `--portal-edge`. A stylesheet rule that sets either
// property restyles the ring and leaves the motion alone.
function portalSwirl(svg, cx, cy, cell, look, inward, spin, at, duration, under) {
  const ring = document.createElementNS(SVGNS, 'g');
  ring.setAttribute('data-journey', inward ? 'swirl-in' : 'swirl-out');
  ring.setAttribute('class', 'portal-swirl');
  ring.setAttribute('data-swirl', inward ? 'in' : 'out');
  ring.setAttribute('data-portal', look.mark);
  ring.setAttribute('data-pair', look.pair);
  ring.setAttribute('data-end', look.end);
  ring.setAttribute('pointer-events', 'none');
  const dash = `stroke-dasharray="${cell * 0.17} ${cell * 0.13}" stroke-linecap="round"`;
  ring.innerHTML =
    `<circle cx="${cx}" cy="${cy}" r="${cell * 0.4}" fill="none"
      stroke="var(--swirl-edge, var(--portal-edge))" stroke-width="${cell * 0.1}" ${dash}/>`
    + `<circle cx="${cx}" cy="${cy}" r="${cell * 0.4}" fill="none"
      stroke="var(--swirl-ink, ${look.ink})" stroke-width="${cell * 0.055}" ${dash}/>`;
  ring.style.opacity = '0';
  ring.style.transformOrigin = `${cx}px ${cy}px`;
  if (under && under.parentNode === svg) svg.insertBefore(ring, under);
  else svg.appendChild(ring);
  const turn = deg => `rotate(${spin * deg}deg)`;
  const frames = inward
    ? [{ opacity: 0, transform: `${turn(0)} scale(1.45)` },
       { opacity: 0.95, transform: `${turn(110)} scale(1)`, offset: 0.35 },
       { opacity: 0, transform: `${turn(320)} scale(0.15)` }]
    : [{ opacity: 0, transform: `${turn(0)} scale(0.2)` },
       { opacity: 0.95, transform: `${turn(120)} scale(1)`, offset: 0.3 },
       { opacity: 0, transform: `${turn(300)} scale(1.55)` }];
  const run = ring.animate(frames, { duration: duration, delay: at, fill: 'forwards',
                                     easing: inward ? 'cubic-bezier(.4,0,.9,.5)'
                                                    : 'cubic-bezier(.1,.5,.3,1)' });
  run.onfinish = () => ring.remove();
  return at + duration;
}

// The end itself reacting: a gulp as the piece goes down into it, a kick as one comes up
// out of it. The glyph the board drew, scaled about its own centre and given back
// untouched (no fill), so the end at rest is the end the drawing made.
function portalPulse(svg, square, cell, view, frames, at, duration) {
  const glyph = svg.querySelector(`[data-piece="${square}"]`);
  if (!glyph || typeof glyph.animate !== 'function') return 0;
  const [x, y] = squareXY(square, cell, view);
  glyph.style.transformOrigin = `${x + cell / 2}px ${y + cell / 2}px`;
  const run = glyph.animate(frames, { duration: duration, delay: at, easing: 'ease-in-out' });
  run.onfinish = () => { glyph.style.transformOrigin = ''; };
  return at + duration;
}
// A move through a portal as one journey (Q836). A ghost of the piece, drawn on the end it
// goes in by, slides there from where it stood, squashes against the rim, and shrinks and
// spins down into it while a swirl in the pair's colour closes over it and the end gulps.
// The piece the board already drew where it lands is held unseen at the other end until
// then, and comes up out of it spinning the same way, overshooting its size, hops the one
// step on, overshoots that and settles with a small squash, while that end kicks and a
// swirl opens round it. Every movement eases out past its mark and back (springy
// cubic-beziers, a keyframe of overshoot and squash), so it reads as a body thrown
// through a hole rather than a slide show. Returns when it lands and the blow a piece
// taken there is struck with: the moment the piece arrives on its square at full size.
//
// The journey itself, drawn: `trip` says where on the drawing (`ex`, `ey` the end it goes
// in by, `fx`, `fy` where it stood, `tx`, `ty` where it lands), the slide in and the step
// out (`idx`, `idy`, `odx`, `ody`), whether the slide lifts (`arc`) or crosses a seam
// (`wrapped`), the two ends' squares (`portal`, `portalOut`) and looks (`lookIn`,
// `lookOut`, as `portalLook` says them). The caller reads all of it off the move.
function portalFlight(svg, node, piece, trip, cell, view) {
  const { ex, ey, fx, fy, tx, ty, idx, idy, odx, ody, arc } = trip;
  const plan = portalJourney(Math.hypot(idx, idy));
  // Which way it turns: clockwise going right (or down), as a wheel rolling that way
  // would, and the same way coming out as going in, so the two halves are one spin.
  const spin = (Math.abs(idx) >= Math.abs(idy) ? idx : idy) < 0 ? -1 : 1;
  const turn = deg => `rotate(${spin * deg}deg)`;
  const ecx = ex + cell / 2, ecy = ey + cell / 2;

  if (piece) {
    const goer = ghostAt(svg, piece, ex, ey, cell);
    goer.setAttribute('data-journey', 'approach');
    portalSlide(goer, -idx, -idy, arc, plan.approach.for);
    if (trip.wrapped) {
      // Across the seam: the copy leaving by the near edge makes the same slide, and the
      // two hand over in the middle, as every crossing does (`animateCrossing`).
      goer.animate(crossOpacity(false), { duration: plan.approach.for, fill: 'backwards' });
      const away = ghostAt(svg, piece, fx + idx, fy + idy, cell);
      portalSlide(away, -idx, -idy, arc, plan.approach.for);
      const fade = away.animate(crossOpacity(true),
                                { duration: plan.approach.for, fill: 'both' });
      fade.onfinish = () => away.remove();
    }
    // The dive, on the drawing inside the ghost, about the end's centre: a squash against
    // the rim and a turn back to wind up, then down and round, gone at the end.
    const inner = goer.firstElementChild || goer;
    if (inner !== goer) inner.setAttribute('data-journey', 'dive');
    inner.style.transformOrigin = `${ecx}px ${ecy}px`;
    const dive = inner.animate([
      { opacity: 1, transform: `${turn(0)} scale(1, 1)`, easing: 'cubic-bezier(.3,.6,.5,1)' },
      { opacity: 1, transform: `${turn(-14)} scale(1.14, 0.86)`, offset: 0.22,
        easing: 'cubic-bezier(.5,0,.85,.45)' },
      { opacity: 1, transform: `${turn(150)} scale(0.55)`, offset: 0.6,
        easing: 'cubic-bezier(.5,0,1,.6)' },
      { opacity: 0, transform: `${turn(400)} scale(0.02)` },
    ], { duration: plan.dive.for, delay: plan.dive.at, fill: 'forwards' });
    dive.onfinish = () => goer.remove();
    portalSwirl(svg, ecx, ecy, cell, trip.lookIn, true, spin,
                plan.dive.at, plan.dive.for + 60, goer);
  }
  portalPulse(svg, trip.portal, cell, view, [
    { transform: 'scale(1)' },
    { transform: 'scale(1.12)', offset: 0.3 },
    { transform: 'scale(0.84)', offset: 0.62 },
    { transform: 'scale(1.05)', offset: 0.84 },
    { transform: 'scale(1)' },
  ], plan.dive.at, plan.dive.for + 120);

  // Out of the other end. Where it comes up is the square one step back from where it
  // lands along its own step out, which is the exit end, or that end's copy across the
  // seam when the step out crosses it, so the swirl and the piece meet there.
  const ocx = tx - odx + cell / 2, ocy = ty - ody + cell / 2;
  portalPulse(svg, trip.portalOut, cell, view, [
    { transform: 'scale(1)' },
    { transform: 'scale(0.86)', offset: 0.22 },
    { transform: 'scale(1.16)', offset: 0.5 },
    { transform: 'scale(0.97)', offset: 0.78 },
    { transform: 'scale(1)' },
  ], Math.max(0, plan.emerge.at - 40), 340);
  portalSwirl(svg, ocx, ocy, cell, trip.lookOut, false, spin,
              Math.max(0, plan.emerge.at - 30), 380, node);
  // The piece itself, about its own centre where it lands: held small and unseen at the
  // exit until its moment (`backwards`), popped up past full size while it is still over
  // the end, thrown the step on past its square with a squash across the way it moves,
  // and settled. No fill after: at rest it is exactly what the board drew.
  const across = Math.abs(odx) >= Math.abs(ody);
  const squash = across ? 'scale(0.9, 1.08)' : 'scale(1.08, 0.9)';
  const give = across ? 'scale(1.03, 0.97)' : 'scale(0.97, 1.03)';
  const shift = k => `translate(${-odx * k}px, ${-ody * k}px)`;
  node.style.transformOrigin = `${tx + cell / 2}px ${ty + cell / 2}px`;
  const emerge = node.animate([
    { opacity: 0, transform: `${shift(1)} ${turn(-320)} scale(0.04)`,
      easing: 'cubic-bezier(.2,.75,.35,1)' },
    { opacity: 1, transform: `${shift(0.9)} ${turn(-40)} scale(1.2)`, offset: 0.4,
      easing: 'cubic-bezier(.45,0,.35,1)' },
    { opacity: 1, transform: `${shift(-0.1)} ${turn(8)} ${squash}`, offset: 0.68,
      easing: 'cubic-bezier(.3,0,.4,1)' },
    { opacity: 1, transform: `${shift(0.02)} ${turn(-2)} ${give}`, offset: 0.86,
      easing: 'ease-out' },
    { opacity: 1, transform: `${shift(0)} ${turn(0)} scale(1, 1)` },
  ], { duration: plan.emerge.for, delay: plan.emerge.at, fill: 'backwards' });
  emerge.onfinish = () => { node.style.transformOrigin = ''; };
  return { landed: plan.landed, blow: { at: plan.strike, dx: odx, dy: ody } };
}
