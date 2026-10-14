// Rule cards: the schema's field tree, rendered as controls and read back into the spec.
//
// The field tree comes from `web/cards.py`, which derives it from
// `spec.jsonschema.rule_spec_schema()`. This module is the rendering half and nothing
// else, so a different renderer replaces this file and keeps the tree.
//
// Three layers, and only the first touches the DOM:
//
// - `fieldHtml` walks a field and a value. It returns HTML, and it pushes one *control*
//   per element it draws: a path, a control type and the raw value the element holds.
// - `readControl` reads the same record back off a live element.
// - `buildControls` turns a list of those records into a value, and `mergeGenerated`
//   writes that value over the keys the tree owns and leaves every other key alone.
//
// Every edit goes through `writeRoot`, which rebuilds the edited root from its controls.
// The round trip in `tests/test_designer.py` runs the same build over the controls the
// render recorded, so a control that cannot say what the spec says fails there.

// ------------------------------------------------------------------ rule cards

let PICKING = null,      // { root, path, label }: which square list a board click writes
    RULE_REFUSED = {},   // rule key -> why this draft cannot take it, from the validator
    RULE_TRIED = '',     // the draft the line above was asked about
    CHOSEN_SPELLING = {};  // which alternative a group is showing while it holds nothing

const LIST_CONTROLS = new Set(['squares', 'pieces', 'ints', 'vectors']);
const MARKERS = new Set(['card', 'list', 'map']);

// A list whose items are pairs of numbers is a list of directions, and the move row
// already writes those as `0,1 1,2` in one box. The schema shape says so -- a `list` of
// `ints` -- so this is not a path anybody has to keep in step with the schema.
function isVectorList(field) {
  return field.control === 'list' && field.item && field.item.control === 'ints';
}

function cardModel() {
  return (DESIGN_KIT && DESIGN_KIT.cards) || null;
}

// ------------------------------------------------- what a rule is called, and what it does
//
// Both come from `spec/describe.py` through `GET /api/design`. The schema names every rule
// by its key and describes thirteen of the twenty-eight, so a card built from the schema
// alone could only show `no_progress_draw` and say nothing about it (finding 2.8).

function ruleWords(key) {
  return ((DESIGN_KIT && DESIGN_KIT.ruleWords) || {})[key] || null;
}

function ruleTitle(key, fallback) {
  const words = ruleWords(key);
  return words ? words.title : fallback;
}

function ruleSays(key) {
  const words = ruleWords(key);
  return words ? words.says : '';
}

// The sibling fields of one rule that say the same thing in different ways, keyed by the
// dotted path the card sits at. Ordered so the last option is the one the compiler reads.
function choiceGroups(dotted) {
  return ((DESIGN_KIT && DESIGN_KIT.ruleChoices) || {})[dotted] || [];
}

// Which spelling a value is written in: the last option that appears in it, because that
// is the one the compiler resolves.
//
// A spelling somebody has taken but not yet filled in writes no key at all -- an empty
// rank and an empty list of ranks are both errors about a field nobody has typed into,
// and a card must not arrive holding one -- so the pick is remembered here and the
// value wins the moment it says anything.
function chosenChoice(group, value, at, root, path) {
  for (let index = group.options.length - 1; index >= 0; index--) {
    if (group.options[index].keys.some(key => value && value[key] !== undefined)) {
      return index;
    }
  }
  const held = CHOSEN_SPELLING[spellingKey(root, path, at)];
  return held === undefined ? 0 : held;
}

function spellingKey(root, path, at) {
  return root + '|' + JSON.stringify(path) + '|' + at;
}

// Which fields a root is drawn from, and the object in the draft it writes into.
function rootFields(root) {
  const model = cardModel();
  if (!model) return [];
  const kind = root.split('.')[0];
  return { rules: model.rules, piece: model.piece, move: model.move, place: model.placement }[kind]
    || [];
}

function rootTarget(root) {
  const [kind, a, b] = root.split('.');
  if (kind === 'rules') return (DRAFT.rules = DRAFT.rules || {});
  if (kind === 'piece') return DRAFT.pieces[Number(a)];
  if (kind === 'move') return DRAFT.pieces[Number(a)].moves[Number(b)];
  if (kind === 'place') return DRAFT.setup.placements[Number(a)];
  return null;
}

// ---------------------------------------------------------------- the value side

function getIn(object, path) {
  return path.reduce((at, key) => (at === undefined || at === null ? undefined : at[key]), object);
}

function setIn(object, path, value) {
  let at = object;
  path.forEach((key, i) => {
    if (i === path.length - 1) { at[key] = value; return; }
    if (at[key] === undefined || at[key] === null || typeof at[key] !== 'object') {
      at[key] = typeof path[i + 1] === 'number' ? [] : {};
    }
    at = at[key];
  });
}

function dropIn(object, path) {
  const parent = getIn(object, path.slice(0, -1));
  const key = path[path.length - 1];
  if (Array.isArray(parent)) parent.splice(key, 1);
  else if (parent) delete parent[key];
}

// What one control record says, or undefined for "leave the key out".
function controlValue(control) {
  const raw = control.raw;
  switch (control.control) {
    // A rule a card of its own draws (the portals, P3), held as it stands: its card writes
    // it, and an edit to any other card writes it back unchanged.
    case 'kept': return raw;
    case 'menu': return raw === '' ? undefined : JSON.parse(raw);
    case 'int': case 'number': return String(raw).trim() === '' ? undefined : Number(raw);
    case 'rank':
      if (String(raw).trim() === '') return undefined;
      return /^\d+$/.test(raw.trim()) ? Number(raw.trim()) : raw.trim();
    case 'text': case 'piece': return raw === '' && !control.required ? undefined : raw;
    case 'vectors': {
      // `0,1 1,2`, exactly as the move row takes it. A pair that is not two numbers is
      // half typed, so it is left out rather than written as a direction nobody meant.
      const pairs = String(raw.text || '').trim().split(/\s+/).filter(Boolean)
        .map(pair => pair.split(',').map(Number))
        .filter(pair => pair.length === 2 && pair.every(n => Number.isFinite(n)));
      if (pairs.length) return pairs;
      return raw.empty || control.required ? [] : undefined;
    }
    case 'squares': case 'pieces': case 'ints': {
      const words = String(raw.text || '').split(/[\s,]+/).filter(Boolean);
      if (words.length) return control.control === 'ints' ? words.map(Number) : words;
      return raw.empty || control.required ? [] : undefined;
    }
    default: return undefined;
  }
}

function buildControls(controls) {
  const out = {};
  for (const control of controls) {
    if (MARKERS.has(control.control)) {
      if (getIn(out, control.path) === undefined) {
        setIn(out, control.path, control.control === 'list' ? [] : {});
      }
      continue;
    }
    const value = controlValue(control);
    if (value !== undefined) setIn(out, control.path, value);
  }
  return out;
}

// Writes `built` over the keys `fields` owns. A partial card shares its object with the
// hand-written form, so only its own keys are replaced, and it goes when it empties.
function mergeGenerated(target, built, fields) {
  for (const field of fields) {
    if (field.partial) {
      const sub = Object.assign({}, target[field.key]);
      mergeGenerated(sub, built[field.key] || {}, field.fields);
      if (Object.keys(sub).length) target[field.key] = sub; else delete target[field.key];
    } else if (built[field.key] !== undefined) {
      target[field.key] = built[field.key];
    } else {
      delete target[field.key];
    }
  }
  return target;
}

// What a field holds when it is first added: the first option, the lowest number, the
// first piece. The validator says what still has to be filled in.
function fieldDefault(field, path, root) {
  const at = path || [field.key];
  const where = root || 'rules';
  const plain = plainDefault(field, at, where);
  const settle = CARD_DEFAULTS[dottedPath(where, at)];
  return settle ? settle(plain) : plain;
}

function plainDefault(field, path, root) {
  const ids = (DRAFT.pieces || []).map(p => p.id);
  switch (field.control) {
    case 'menu': return field.options.length ? field.options[0].value : '';
    case 'int': case 'number': return field.minimum === undefined ? 0 : field.minimum;
    case 'piece': return ids[0] || '';
    case 'text': case 'rank': return '';
    case 'squares': case 'pieces': case 'ints': return [];
    case 'map': return {};
    case 'list': return [fieldDefault(field.item, path.concat([0]), root)];
    case 'either': return fieldDefault(field.branches[0].field, path, root);
    case 'card': {
      const out = {};
      field.fields.filter(f => f.required).forEach(f => {
        out[f.key] = fieldDefault(f, path.concat([f.key]), root);
      });
      return out;
    }
    default: return null;
  }
}

// ------------------------------------------------- what a new card arrives holding

// The tree's own default is the first option and the first piece, and for most fields
// that is right. It is wrong wherever a rule is *about* a piece with a part to play:
// castling arriving pointed at a common piece, a chain link pointing a piece at itself,
// a landing arriving as an empty pair. Each cost an error before the person had typed
// anything (walks 1.1, 1.2 and 1.4).
//
// Each entry takes the plain default and hands back what this draft makes of it. What a
// piece *is* comes from the draft every time -- the royal flag, the opening position,
// the moves already written -- and never from a letter, so a game with no king and no
// corners falls back to the plain default and meets the validator's own sentence about
// the game.
//
// Keyed by the field's path, dotted from the spec's top with `[]` for every item of a
// list, the way `web/cards.PIECE_PATHS` spells it. A rule the schema adds and this table
// does not name arrives at the plain default.
const CARD_DEFAULTS = {
  'rules.castling': card => Object.assign({}, card, {
    king: royalId() || card.king,
    rook: cornerId(royalId()) || card.rook,
  }),
  'rules.promotion': card => Object.assign({}, card, {
    to: card.to && card.to.length ? card.to : [promotionTarget(card.piece)].filter(Boolean),
  }),
  'rules.promotion[]': card => CARD_DEFAULTS['rules.promotion'](card),
  'rules.obstacles': card => Object.assign({}, card, { squares: [openSquare(false)] }),
  'rules.reach': card => Object.assign({}, card, { squares: [openSquare(true)] }),
  // A second wild piece is most often another of the first one's kind (the wolves walk,
  // step 13), so it arrives as the entry above it, on a square nothing names yet.
  'rules.npcs[]': card => Object.assign({}, card, { route: [openSquare(false)] },
    lastWildPiece() ? { piece: lastWildPiece() } : {}),
  // A link starts at a piece a strike reaches, which a royal piece is not (it is immune),
  // and ends at a piece that is not royal, for the reason a promotion does.
  'rules.strike.chain[]': link => {
    const royal = royalId();
    const from = link.from === royal ? (pieceIds().find(id => id !== royal) || link.from)
      : link.from;
    return Object.assign({}, link, { from, to: promotionTarget(from) });
  },
  'rules.strike.landing': () => landingFor(['strike']),
  'rules.contacts.landing': () => landingFor(['contacts']),
  // The link below leaves what it moved onto standing, and the rule's own default puts
  // the mover on that square, so the card has to say where the survivor goes. Royalty
  // is left out of it: a link turns what it lands on into something else, and a royal
  // piece that is turned rather than taken is a royal piece nobody can win against.
  'rules.contacts': card => Object.assign({}, card,
    { position: card.position || 'stay' },
    royalId() ? { immune: [royalId()] } : {}),
  'rules.contacts.links[]': link => {
    // A link that says neither what the mover becomes nor what it moves onto becomes is
    // an ordinary capture written the long way, and the validator says so.
    const onto = otherId(link.mover);
    const becomes = otherId(onto);
    return Object.assign({}, link, { onto },
      becomes === onto ? {} : { onto_to: becomes });
  },
};

// A part added inside a rule can need the rule around it to say something else: a chain
// under the strike's default effect `capture` is a chain the core never plays, and the
// validator refuses it (Q538, `schema.strike_states_nothing`). Each entry takes the card
// that holds the new part and sets what the part needs of it, and leaves alone anything
// somebody already chose. Keyed like `CARD_DEFAULTS`, by the part's own path.
const HOLDER_SETTLES = {
  // A struck piece that survives turns into its link's target: the effect is the
  // schema's first one that is not `capture`, and the attacker stays where it stood,
  // since under `advance` it would take the square the survivor still holds.
  'rules.strike.chain': (strike, path) => {
    if ((strike.effect || 'capture') === 'capture') {
      const effect = fieldAt('rules', path.slice(0, -1).concat(['effect']));
      const other = effect && (effect.options || []).find(o => o.value !== 'capture');
      if (other) strike.effect = other.value;
    }
    if ((strike.position || 'advance') === 'advance') strike.position = 'stay';
    // A royal piece that is turned rather than taken is one nobody can win against, and
    // the validator asks for every royal to be named immune (as `rules.contacts` does).
    const royal = royalId();
    const immune = Array.isArray(strike.immune) ? strike.immune : [];
    if (royal && !immune.includes(royal)) strike.immune = immune.concat([royal]);
  },
};

// The *add it* button of an optional part: the part's own default, then whatever the
// card holding it has to say for the part to be a rule the core plays.
function addPart(target, root, path) {
  const field = fieldAt(root, path);
  setIn(target, path, field ? fieldDefault(field, path, root) : {});
  const settle = HOLDER_SETTLES[dottedPath(root, path)];
  const holder = getIn(target, path.slice(0, -1));
  if (settle && holder && typeof holder === 'object') settle(holder, path);
}

const ROOT_PATHS = { rules: 'rules', piece: 'pieces[]', move: 'pieces[].moves[]',
                     place: 'setup.placements[]' };

function dottedPath(root, path) {
  let out = ROOT_PATHS[root.split('.')[0]] || root;
  for (const key of path) out += typeof key === 'number' ? '[]' : '.' + key;
  return out;
}

function pieceIds() {
  return (DRAFT.pieces || []).map(p => p.id);
}

// The piece of the draft's last wild piece, while that is still one of its pieces.
function lastWildPiece() {
  const npcs = (DRAFT.rules || {}).npcs;
  const last = Array.isArray(npcs) && npcs.length ? npcs[npcs.length - 1] : null;
  const id = last && typeof last.piece === 'string' ? last.piece : '';
  return pieceIds().includes(id) ? id : '';
}

// The piece the draft marks royal, if it marks one.
function royalId() {
  const royal = (DRAFT.pieces || []).find(p => p.royal);
  return royal ? royal.id : '';
}

// Any piece other than `id`, or `id` again when the draft has only the one.
function otherId(id) {
  const ids = pieceIds();
  return ids.find(other => other !== id) || id || ids[0] || '';
}

// What a promoting piece becomes: not itself, because that is no promotion, and not the
// royal piece, because a second of those changes the game rather than the piece.
function promotionTarget(from) {
  const ids = pieceIds(), royal = royalId();
  return ids.find(id => id !== from && id !== royal) || ids.find(id => id !== from)
    || from || '';
}

// The piece standing on a corner of the opening position. Castling's other piece starts
// in a corner wherever the rule is played, and which piece that is is the draft's to say.
function cornerId(besides) {
  const preview = lastPreview();
  if (!preview || !preview.board) return '';
  const W = preview.width, H = preview.height;
  for (const square of [0, W - 1, (H - 1) * W, H * W - 1]) {
    const id = previewPiece(preview, square);
    if (id && id !== besides) return id;
  }
  return '';
}

function previewPiece(preview, square) {
  const value = preview.board[square];
  if (!value) return '';
  return (preview.pieceIds || [])[(value - 1) % preview.nTypes] || '';
}

// A square with nothing on it at the start and nothing already said about it, for the
// lists that want ground rather than a piece. `far` walks back from the last square,
// which is the corner a race is usually run to; otherwise from the first. The opening
// position answers, so a mirrored setup's far side counts as occupied.
function openSquare(far) {
  const preview = lastPreview();
  if (!preview || !preview.board) return far ? 'a1' : 'b2';
  // A hexagon's absent corners (schema 16) are no ground at all, and travel apart from
  // the walls since Q281b.
  const W = preview.width;
  const blocked = new Set((preview.obstacles || []).concat(preview.absent || []));
  const spoken = squaresNamed(DRAFT.rules || {});
  const squares = preview.board.map((_, square) => square);
  if (far) squares.reverse();
  const free = squares.find(square => !preview.board[square] && !blocked.has(square)
    && !spoken.has(squareName(square % W, Math.floor(square / W))));
  const square = free === undefined ? (far ? preview.board.length - 1 : 0) : free;
  return squareName(square % W, Math.floor(square / W));
}

// Every square the rules already name, wherever they name it. Two wild pieces may not
// walk the same square and an obstacle under a goal is a goal nobody reaches, so a
// second square list starts where the first one left off.
function squaresNamed(value, found) {
  const out = found || new Set();
  if (typeof value === 'string') {
    if (/^[a-z]\d+$/.test(value)) out.add(value);
  } else if (Array.isArray(value)) {
    value.forEach(item => squaresNamed(item, out));
  } else if (value && typeof value === 'object') {
    Object.keys(value).forEach(key => squaresNamed(value[key], out));
  }
  return out;
}

// A landing says where a struck or displaced piece goes, and the validator only lets it
// mean anything where the taker is given the choice -- so adding one says that is what
// was wanted, on the rule the landing belongs to.
function landingFor(path) {
  const card = getIn(DRAFT.rules || {}, path);
  if (card) card.position = 'choose';
  return stepOfRoyalty();
}

// A landing puts a struck piece one square from where it stood, and what "one square"
// is depends on the lattice. Rather than work that out here -- geometry belongs to the
// compiler -- it is read off a step the draft already writes: the royal piece's, which
// is the king's step wherever there is a king, and the first piece's otherwise.
function stepOfRoyalty() {
  const pieces = DRAFT.pieces || [];
  const stepper = pieces.find(p => p.royal && (p.moves || []).length)
    || pieces.find(p => (p.moves || []).length);
  const move = stepper && stepper.moves[0];
  const vectors = move && Array.isArray(move.vectors) && move.vectors.length
    ? move.vectors.map(pair => pair.slice()) : [[1, 0]];
  return move && move.symmetry ? { vectors, symmetry: move.symmetry } : { vectors };
}

// The branch of an `either` a value is written in.
function branchOf(field, value) {
  const fits = f => {
    if (value === undefined) return false;
    if (Array.isArray(value)) return f.control === 'list' || LIST_CONTROLS.has(f.control);
    if (value !== null && typeof value === 'object') return f.control === 'card' || f.control === 'map';
    if (f.control === 'menu') return f.options.some(o => o.value === value);
    return !LIST_CONTROLS.has(f.control) && !MARKERS.has(f.control);
  };
  const found = field.branches.findIndex(b => fits(b.field));
  return found < 0 ? 0 : found;
}

// ---------------------------------------------------------------- the HTML side

// Paths and menu values are JSON, so a quote is ordinary in these attributes. `esc`
// escapes it.
function cattrs(root, path, control, required) {
  return `data-croot="${esc(root)}" data-cpath="${esc(JSON.stringify(path))}"`
    + ` data-cctl="${control}"${required ? ' data-creq="1"' : ''}`;
}

function saysHtml(text) {
  return text ? `<div class="muted small cardSays">${esc(text)}</div>` : '';
}

// A field and its value as HTML. Every element that holds part of the value is pushed
// onto `out` with the raw value it was drawn with.
function fieldHtml(field, value, path, root, out, head) {
  const push = (control, raw) => out.push({ path, control, raw, required: !!field.required });
  const label = head === undefined ? `<span>${esc(field.label)}</span>` : head;
  const ids = (DRAFT.pieces || []).map(p => p.id);

  if (field.control === 'menu') {
    const options = field.options.slice();
    if (value !== undefined && !options.some(o => o.value === value)) {
      options.push({ value, label: String(value) + ' (not one the rules allow)' });
    }
    const raw = value === undefined ? '' : JSON.stringify(value);
    push('menu', raw);
    const chosen = options.find(o => o.value === value);
    return `<label class="field">${label}
      <select ${cattrs(root, path, 'menu', field.required)}>
        ${field.required && value !== undefined ? '' : '<option value="">not said</option>'}
        ${options.map(o => `<option value="${esc(JSON.stringify(o.value))}"${
          o.value === value ? ' selected' : ''} title="${esc(o.says || '')}">${esc(o.label)}</option>`
        ).join('')}
      </select>
      ${saysHtml(chosen && chosen.says ? chosen.says : field.says)}</label>`;
  }

  if (field.control === 'int' || field.control === 'number' || field.control === 'rank'
      || field.control === 'text') {
    const raw = value === undefined ? '' : String(value);
    push(field.control, raw);
    const numeric = field.control === 'int' || field.control === 'number';
    const bounds = (field.minimum !== undefined ? ` min="${field.minimum}"` : '')
      + (field.maximum !== undefined ? ` max="${field.maximum}"` : '')
      + (field.control === 'number' ? ' step="any"' : '');
    const hint = numeric && (field.minimum !== undefined || field.maximum !== undefined)
      ? ` title="${field.minimum === undefined ? '' : 'from ' + field.minimum}${
          field.maximum === undefined ? '' : ' to ' + field.maximum}"` : '';
    return `<label class="field">${label}
      <input ${cattrs(root, path, field.control, field.required)}${
        numeric ? ' type="number"' + bounds : ''}${hint}
        ${field.control === 'rank' ? ' placeholder="last, or a row number (your own back row is 0)"' : ''}
        value="${esc(raw)}">
      ${saysHtml(field.says)}</label>`;
  }

  if (field.control === 'piece') {
    const raw = value === undefined ? '' : String(value);
    push('piece', raw);
    const choices = ids.slice();
    if (raw && !choices.includes(raw)) choices.push(raw);
    return `<label class="field">${label}
      <select ${cattrs(root, path, 'piece', field.required)}>
        ${field.required && raw ? '' : '<option value="">not said</option>'}
        ${choices.map(id => `<option value="${esc(id)}"${id === raw ? ' selected' : ''}>${
          esc(pieceLabel(id))}</option>`).join('')}
      </select>
      ${saysHtml(field.says)}</label>`;
  }

  if (isVectorList(field)) {
    const list = Array.isArray(value) ? value : [];
    const raw = { text: list.map(pair => (pair || []).join(',')).join(' '),
                  empty: Array.isArray(value) && !list.length };
    push('vectors', raw);
    return `<div class="field">${label}
      <span class="listControl" ${cattrs(root, path, 'vectors', field.required)}>
        <input data-cpart="text" value="${esc(raw.text)}" placeholder="0,1 1,2">
        ${field.required ? '' : `<label class="pick small"><input type="checkbox" data-cpart="empty"${
          raw.empty ? ' checked' : ''}> none</label>`}
      </span>
      ${saysHtml(field.says)}
      <div class="muted small cardSays">One step as sideways,forward &mdash; 0,1 is one
        square forward and 1,0 is one to the right; several separated by spaces. The same
        notation as a move row's.</div></div>`;
  }

  if (LIST_CONTROLS.has(field.control)) {
    const list = Array.isArray(value) ? value : [];
    const raw = { text: list.join(' '), empty: Array.isArray(value) && !list.length };
    push(field.control, raw);
    const picking = PICKING && PICKING.root === root
      && JSON.stringify(PICKING.path) === JSON.stringify(path);
    const helper = field.control === 'squares'
      ? `<button data-cpick="${esc(JSON.stringify(path))}" data-croot-pick="${esc(root)}"
           data-clabel="${esc(field.label)}" class="${picking ? 'on' : ''}"
           title="click squares on the opening board">${picking ? 'done picking' : 'pick on the board'}</button>`
      : field.control === 'pieces'
        ? ids.map(id => `<button class="chip${list.includes(id) ? ' on' : ''}"
            data-ctoggle="${esc(id)}">${esc(id)}</button>`).join('')
        : '';
    const placeholder = { squares: 'a1 b1 c1', pieces: 'piece letters', ints: '0 1 2' }[field.control];
    return `<div class="field">${label}
      <span class="listControl" ${cattrs(root, path, field.control, field.required)}>
        <input data-cpart="text" value="${esc(raw.text)}" placeholder="${placeholder}">
        ${field.required ? '' : `<label class="pick small"><input type="checkbox" data-cpart="empty"${
          raw.empty ? ' checked' : ''}> none</label>`}
        ${helper}
      </span>
      ${saysHtml(field.says)}</div>`;
  }

  if (field.control === 'either') {
    const at = branchOf(field, value);
    const branch = field.branches[at];
    const switcher = `<select data-ceither="${esc(JSON.stringify(path))}" data-cfrom="${at}"
        data-croot-either="${esc(root)}">
      ${field.branches.map((b, i) => `<option value="${i}"${i === at ? ' selected' : ''}>${
        esc(b.label)}</option>`).join('')}</select>`;
    const inner = fieldHtml(Object.assign({}, branch.field, { label: field.label, says: '' }),
      value, path, root, out, '');
    return `<div class="either"><div class="row small eitherHead">${label} ${switcher}</div>
      ${saysHtml(field.says)}${inner}</div>`;
  }

  if (field.control === 'card') {
    if (value === undefined || value === null || typeof value !== 'object') {
      return `<div class="field">${label}
        <button data-cadd="${esc(JSON.stringify(path))}" data-croot-add="${esc(root)}">add ${
          esc(field.label)}</button>${saysHtml(field.says)}</div>`;
    }
    out.push({ path, control: 'card', raw: null, required: true });
    const marker = `<i hidden ${cattrs(root, path, 'card', true)}></i>`;
    const body = cardBodyHtml(field, value, path, root, out);
    const drop = field.required ? '' : `<button class="cardDrop" data-cdel="${
      esc(JSON.stringify(path))}" data-croot-del="${esc(root)}" title="remove">&times;</button>`;
    return `<div class="subCard">${marker}<div class="cardHead">${label}${drop}</div>
      ${saysHtml(field.says)}<div class="cardBody">${body}</div></div>`;
  }

  if (field.control === 'list') {
    const list = Array.isArray(value) ? value : [];
    if (!Array.isArray(value) && !field.required) {
      return `<div class="field">${label}
        <button data-cadd="${esc(JSON.stringify(path))}" data-croot-add="${esc(root)}">add ${
          esc(field.label)}</button>${saysHtml(field.says)}</div>`;
    }
    out.push({ path, control: 'list', raw: null, required: true });
    const items = list.map((item, i) => {
      const itemPath = path.concat([i]);
      const card = fieldHtml(Object.assign({}, field.item, { required: true }), item, itemPath,
        root, out, `<span>${esc(field.label)} ${i + 1}</span>`);
      return card.replace('<div class="cardHead">', `<div class="cardHead"><button class="cardDrop"
        data-cdel="${esc(JSON.stringify(itemPath))}" data-croot-del="${esc(root)}"
        title="remove this one">&times;</button>`);
    }).join('');
    return `<div class="field">${label}${saysHtml(field.says)}
      <i hidden ${cattrs(root, path, 'list', true)}></i>
      ${items}
      <button data-cpush="${esc(JSON.stringify(path))}" data-croot-push="${esc(root)}">add one</button>
    </div>`;
  }

  if (field.control === 'map') {
    const entries = value && typeof value === 'object' ? Object.keys(value) : [];
    if (!(value && typeof value === 'object') && !field.required) {
      return `<div class="field">${label}
        <button data-cadd="${esc(JSON.stringify(path))}" data-croot-add="${esc(root)}">add ${
          esc(field.label)}</button>${saysHtml(field.says)}</div>`;
    }
    out.push({ path, control: 'map', raw: null, required: true });
    const rows = entries.map(key => `<div class="mapRow">
      <b>${esc(pieceLabel(key))}</b>
      ${fieldHtml(Object.assign({}, field.value, { says: '' }), value[key], path.concat([key]),
        root, out, '')}
      <button class="cardDrop" data-cdel="${esc(JSON.stringify(path.concat([key])))}"
        data-croot-del="${esc(root)}" title="remove">&times;</button>
    </div>`).join('');
    const free = ids.filter(id => !entries.includes(id));
    const adder = free.length ? `<select data-cmapadd="${esc(JSON.stringify(path))}"
        data-croot-map="${esc(root)}"><option value="">add a piece...</option>${
        free.map(id => `<option value="${esc(id)}">${esc(pieceLabel(id))}</option>`).join('')
      }</select>` : '';
    return `<div class="field">${label}${saysHtml(field.says)}
      <i hidden ${cattrs(root, path, 'map', true)}></i>${rows}${adder}</div>`;
  }
  return '';
}

// A card's fields, with each group of alternative spellings drawn as one switcher and
// one spelling. A field in no group is drawn where it stands, and a group is drawn where
// its first field stood, so the card keeps the schema's order either way.
function cardBodyHtml(field, value, path, root, out) {
  const dotted = dottedPath(root, path);
  const hidden = otherShapesKeys(dotted, value);
  const shown = field.fields.filter(f => !hidden.has(f.key));
  const groups = choiceGroups(dotted);
  if (!groups.length) {
    return shown.map(f => fieldHtml(f, value[f.key], path.concat([f.key]), root, out))
      .join('');
  }
  const groupOf = {};
  groups.forEach((group, at) => {
    group.options.forEach(option => option.keys.forEach(key => { groupOf[key] = at; }));
  });
  const drawn = new Set();
  return shown.map(f => {
    const at = groupOf[f.key];
    if (at === undefined) {
      return fieldHtml(f, value[f.key], path.concat([f.key]), root, out);
    }
    if (drawn.has(at)) return '';
    drawn.add(at);
    return choiceHtml(groups[at], at, shown, value, path, root, out);
  }).join('');
}

// The draft's board shape, as `board.shape` states it: a hexagon, or the grid every board
// was before schema 16.
function draftShape() {
  return ((DRAFT && DRAFT.board) || {}).shape === 'hexagon' ? 'hexagon' : 'grid';
}

// The keys of the card at `dotted` that belong to another board shape and that the card
// does not hold, which are left off it: a hexagon's wrap card is its three pairs of sides
// and a grid's is its two axes. A key the card does hold is drawn whatever the shape, so
// what the draft says is on screen beside the validator's refusal of it. Which keys are
// whose is the server's (`web/cards.shape_keys`).
function otherShapesKeys(dotted, value) {
  const byShape = ((cardModel() || {}).shapeKeys || {})[dotted];
  const out = new Set();
  if (!byShape) return out;
  const mine = new Set(byShape[draftShape()] || []);
  Object.values(byShape).forEach(keys => keys.forEach(key => {
    if (!mine.has(key) && !(value && value[key] !== undefined)) out.add(key);
  }));
  return out;
}

// One group: a menu of the spellings, and the fields of the one in use. The others are
// not on screen, so a promotion zone is one control rather than three with an *empty
// list* tick beside each.
function choiceHtml(group, at, fields, value, path, root, out) {
  const chosen = chosenChoice(group, value, at, root, path);
  const switcher = `<select data-cchoice="${esc(JSON.stringify(path))}"
      data-cgroup="${at}" data-croot-choice="${esc(root)}">
    ${group.options.map((option, index) => `<option value="${index}"${
      index === chosen ? ' selected' : ''}>${esc(option.label)}</option>`).join('')}</select>`;
  const inner = group.options[chosen].keys.map(key => {
    const f = fields.find(one => one.key === key);
    return f ? fieldHtml(f, value[key], path.concat([key]), root, out) : '';
  }).join('');
  return `<div class="either"><div class="row small eitherHead">
    <span>${esc(group.label)}</span> ${switcher}</div>${inner}</div>`;
}

// A spelling taken in place of the one in use: the group's other keys go, the chosen
// one's arrive at their defaults, and the rule's own settle then fills what it knows
// about this draft. The settle can name a key from another option -- promotion's fills
// `to` whichever spelling is showing -- so the others are cleared again after it.
function switchChoice(root, path, groupAt, optionAt) {
  const target = rootTarget(root);
  const group = choiceGroups(dottedPath(root, path))[groupAt];
  if (!target || !group) return;
  // A promotion is one rule or a list of them, so the field at this path is the choice
  // between those two shapes; the card is the branch the value is written in.
  let field = fieldAt(root, path);
  if (field && field.control === 'either') {
    field = field.branches[branchOf(field, getIn(target, path))].field;
  }
  if (!field || field.control !== 'card') return;
  const option = group.options[optionAt];
  if (!option) return;
  CHOSEN_SPELLING[spellingKey(root, path, groupAt)] = optionAt;
  structural(root, holder => {
    const card = getIn(holder, path);
    if (!card) return;
    group.options.forEach(other => other.keys.forEach(key => { delete card[key]; }));
    option.keys.forEach(key => {
      const sub = field.fields.find(one => one.key === key);
      if (!sub) return;
      const value = fieldDefault(sub, path.concat([key]), root);
      // A default that says nothing is the same as leaving the key out, and an empty
      // rank or an empty list of them is an error about a field nobody has typed into
      // yet. The spelling being shown is remembered rather than written.
      if (value === '' || (Array.isArray(value) && !value.length)) return;
      card[key] = value;
    });
    const settle = CARD_DEFAULTS[dottedPath(root, path)];
    if (settle) Object.assign(card, settle(card));
    group.options.forEach(other => {
      if (other !== option) other.keys.forEach(key => { delete card[key]; });
    });
    // Kept here, with the draft as it now stands: a spelling that writes no key can
    // leave the draft unchanged, and an unchanged draft is not kept again
    // (`snapshotDraft`), so a reload would lose the pick (Q541).
    if (typeof keepDraft === 'function') keepDraft();
  });
}

function pieceLabel(id) {
  const piece = (DRAFT.pieces || []).find(p => p.id === id);
  return piece && piece.name ? `${id} (${piece.name})` : String(id);
}

// Every generated field of one root, as HTML plus the control records it drew.
function rootHtml(root, target, out) {
  return rootFields(root).map(field => fieldHtml(field, (target || {})[field.key], [field.key], root,
    out)).join('');
}

// The rules as cards: one per rule the draft states, most used first, and a menu of
// the rest in the same order with the rules nothing in the library uses at the bottom.
function rulesCardsHtml(out) {
  const model = cardModel();
  if (!model) return '<div class="muted small">loading the rules...</div>';
  const rules = DRAFT.rules || {};
  const byKey = Object.fromEntries(model.rules.map(f => [f.key, f]));
  const cards = model.ruleOrder.filter(key => key in rules).map(key => {
    // The portals have a card of their own, in sentences, one per pair (P3, D4); the
    // generated one stays reachable as the game's text.
    if (key === 'portals' && Array.isArray(rules.portals)) return portalCardsHtml(out);
    const fixed = model.fixed.includes(key);
    // One paragraph says what the rule is (the wolves walk, step 4): the rule's own words
    // where it has them. A list, card or map rule's description is about the whole rule
    // too, so beside those words it is the same thing said twice, and so is any
    // description that is those words again. A choice's description, which says what
    // the choices mean, stays.
    const field = ruleSaysTwice(key, byKey[key])
      ? Object.assign({}, byKey[key], { says: '' }) : byKey[key];
    const inner = fieldHtml(field, rules[key], [key], 'rules', out, '');
    return `<div class="ruleCard" data-rule="${esc(key)}">
      <div class="cardHead"><b>${esc(ruleTitle(key, field.label))}</b>${fixed ? '' : `<button class="cardShow"
        data-cshow="${esc(key)}" title="the engine plays games of this draft and stops where this rule changes one">show me</button><button class="cardDrop"
        data-cdel="${esc(JSON.stringify([key]))}" data-croot-del="rules"
        title="remove this rule">&times;</button>`}</div>
      ${saysHtml(ruleSays(key))}
      ${field.control === 'card' || field.control === 'list' || field.control === 'map'
        ? inner.replace('<div class="cardHead"></div>', '') : inner}
    </div>`;
  }).join('');
  const unknown = Object.keys(rules).filter(key => !(key in byKey));
  return `${cards}
    ${unknown.length ? `<div class="error">This game uses ${unknown.length === 1
      ? 'a rule' : 'rules'} the editor doesn't know (${esc(unknown.join(', '))}); only the
      game's text below can change ${unknown.length === 1 ? 'it' : 'them'}.</div>` : ''}
    <div id="dRuleMenu">${ruleMenuHtml()}</div>`;
}

function ruleSaysTwice(key, field) {
  const said = ruleSays(key);
  if (!said || !field || !field.says) return false;
  if (['list', 'card', 'map'].includes(field.control)) return true;
  const bare = text => String(text).toLowerCase().replace(/[^a-z0-9]+/g, '');
  return bare(said) === bare(field.says);
}

// The menu of rules not in the draft, and the ones that cannot go in it. Its own function
// because the refusals arrive after a check and only this much is redrawn for them -- a
// redraw of the whole panel would close a card somebody is filling in.
function ruleMenuHtml() {
  const model = cardModel();
  if (!model) return '';
  const rules = DRAFT.rules || {};
  const byKey = Object.fromEntries(model.rules.map(f => [f.key, f]));
  const absent = model.ruleOrder.filter(key => !(key in rules));
  const option = key => `<option value="${esc(key)}" title="${esc(ruleSays(key))}">${
    esc(ruleTitle(key, byKey[key].label))} (${model.ruleCounts[key]} in the library)</option>`;
  const open = absent.filter(key => !RULE_REFUSED[key]);
  const common = open.filter(key => !model.rare.includes(key));
  const rare = open.filter(key => model.rare.includes(key));
  const shut = absent.filter(key => RULE_REFUSED[key]);
  return `<label class="field" style="margin-top:10px"><span>add a rule</span>
      <select id="dAddRule"><option value="">choose one...</option>
        ${common.map(option).join('')}
        ${rare.length ? `<optgroup label="rarely used">${rare.map(option).join('')}</optgroup>` : ''}
      </select></label>
    <div class="muted small" id="dAddRuleSays"></div>
    <button id="dAddRuleGo" style="margin-top:6px" disabled>add it</button>
    ${refusedRulesHtml(shut, byKey)}`;
}

// The rules this draft cannot take, and the validator's reason for each (option A9).
// Out of the menu rather than greyed inside it, because a disabled option cannot be
// chosen and a reason nobody can reach is a reason nobody reads. The wolf cost eight
// actions before this sentence arrived (finding 2.10); here it arrives before the first.
function refusedRulesHtml(shut, byKey) {
  if (!shut.length) return '';
  // Asked only of a draft that is a game, so while one is half made these are the last
  // answer rather than this draft's -- said out loud, as the dimmed board says it.
  const stale = !!(DRAFT_CHECK && !DRAFT_CHECK.valid);
  return `<details class="panel refusedRules" style="margin-top:10px">
    <summary class="muted small">${shut.length} rule${shut.length === 1 ? '' : 's'}
      ${shut.length === 1 ? 'does' : 'do'} not go in this game</summary>
    ${stale ? `<div class="muted small">This list is from the last time the draft was a
      complete game. It is asked again as soon as the draft is one.</div>` : ''}
    <ul class="refusedList">${shut.map(key => `<li>
      <b>${esc(ruleTitle(key, byKey[key].label))}</b>
      <span class="muted small">${esc(RULE_REFUSED[key])}</span></li>`).join('')}</ul>
  </details>`;
}

// ------------------------------------------------- rules that do not go in this game
//
// Option A9. Every rule the menu offers is tried against the validator with the card the
// form would actually add, and the ones it refuses come out of the menu with its reason
// beside them. The cards are built here, because what a card arrives holding is the
// form's answer and depends on the draft; whether it is a game is the validator's.

// What each absent rule would arrive as. The same `fieldDefault` the menu's *add it*
// uses, so what is judged is what would be added.
function ruleCandidates() {
  const model = cardModel();
  const rules = DRAFT.rules || {};
  if (!model) return {};
  const out = {};
  model.ruleOrder.filter(key => !(key in rules) && !model.fixed.includes(key))
    .forEach(key => {
      const field = model.rules.find(f => f.key === key);
      if (field) out[key] = fieldDefault(field, [key], 'rules');
    });
  return out;
}

// Asked after a check has come back, because a card reads the opening position for the
// piece it needs and a draft that is not a game has none. Asked once per draft: the
// answer is about the rules that are *not* in it, which a keystroke in a name does not
// move, so the draft it was asked about is remembered.
function loadRefusals() {
  const candidates = ruleCandidates();
  // Everything a refusal can turn on, and the cards themselves. The name and the
  // description are left out, so typing a sentence about the game asks nothing.
  const asking = JSON.stringify([DRAFT.board, DRAFT.pieces, DRAFT.setup, DRAFT.rules,
                                 candidates]);
  if (asking === RULE_TRIED) return;
  RULE_TRIED = asking;
  api('/api/design/rules', { spec: DRAFT, candidates }).then(answer => {
    if (RULE_TRIED !== asking) return;
    RULE_REFUSED = answer.refused || {};
    redrawRuleMenu();
  }).catch(() => {
    // A menu that offers everything is the behaviour before this existed, and a
    // refusal nobody could fetch must not stop a rule being tried.
    RULE_REFUSED = {};
  });
}

// Only the menu and the refused list. Redrawing the form would close a card somebody is
// filling in while the answer was still in the air.
function redrawRuleMenu() {
  const form = document.getElementById('form');
  const box = form && form.querySelector('#dRuleMenu');
  if (!box) return;
  // A rule somebody has chosen and not yet added survives the redraw. The answer lands a
  // second or so after any change, which is when a person is choosing: the menu used to
  // come back empty with *add it* switched off, so the first press did nothing (Q489).
  const held = box.querySelector('#dAddRule');
  const chosen = held ? held.value : '';
  box.innerHTML = ruleMenuHtml();
  // The box and nothing else: `wireCards` on the whole form would stack a second
  // listener on every folded extras block each time an answer came back.
  wireCards(box);
  const menu = box.querySelector('#dAddRule');
  if (chosen && menu && [...menu.options].some(option => option.value === chosen)) {
    menu.value = chosen;
    if (menu.onchange) menu.onchange();
  }
}

// The generated part of one piece, one move or one placement, folded away unless it
// already says something.
function extrasHtml(root, target, summary) {
  const fields = rootFields(root);
  if (!fields.length) return '';
  const said = fields.some(f => target && target[f.key] !== undefined
    && !(f.partial && Object.keys(target[f.key]).every(k => !f.fields.some(g => g.key === k))));
  const out = [];
  return `<details class="extras"${said ? ' open' : ''}><summary class="muted small">${
    esc(summary)}</summary><div class="cardBody">${rootHtml(root, target, out)}</div></details>`;
}

// ---------------------------------------------------------------- reading and wiring

function readControl(node) {
  const control = node.dataset.cctl;
  const path = JSON.parse(node.dataset.cpath);
  const required = node.dataset.creq === '1';
  if (control === 'kept') {
    return { path, control, required, raw: getIn(rootTarget(node.dataset.croot) || {}, path) };
  }
  if (MARKERS.has(control)) return { path, control, raw: null, required };
  if (LIST_CONTROLS.has(control)) {
    const text = node.querySelector('[data-cpart="text"]');
    const empty = node.querySelector('[data-cpart="empty"]');
    return { path, control, required, raw: { text: text ? text.value : '', empty: !!(empty && empty.checked) } };
  }
  return { path, control, raw: node.value, required };
}

// Rebuilds one root from every control drawn for it, and writes it into the draft.
function writeRoot(root) {
  const form = document.getElementById('form');
  const target = rootTarget(root);
  if (!form || !target) return;
  const nodes = [...form.querySelectorAll('[data-cpath]')].filter(n => n.dataset.croot === root);
  mergeGenerated(target, buildControls(nodes.map(readControl)), rootFields(root));
  checkDraft();
  if (root.startsWith('move') || root.startsWith('piece')) loadMoveMap();
}

function structural(root, change) {
  const target = rootTarget(root);
  if (!target) return;
  change(target);
  drawDesigner();
  checkDraft();
}

// The field a path names under a root, for the defaults an add needs.
function fieldAt(root, path) {
  let field = { control: 'card', fields: rootFields(root) };
  for (const key of path) {
    // An index goes into a list branch, a name into a card or a map.
    if (field.control === 'either') {
      const want = typeof key === 'number' ? ['list'] : ['card', 'map'];
      const branch = field.branches.find(b => want.includes(b.field.control));
      if (!branch) return null;
      field = branch.field;
    }
    if (field.control === 'list') field = field.item;
    else if (field.control === 'map') field = field.value;
    else if (field.control === 'card') field = field.fields.find(f => f.key === key) || null;
    else return null;
    if (!field) return null;
  }
  return field;
}

function wireCards(form) {
  // A click on a piece's card selects the piece and redraws the form, which would close
  // a menu inside its extras as it opens.
  form.querySelectorAll('details.extras').forEach(node => {
    node.addEventListener('click', event => event.stopPropagation());
  });
  form.querySelectorAll('[data-cpath]').forEach(node => {
    if (MARKERS.has(node.dataset.cctl) || node.dataset.cctl === 'kept') return;
    const write = () => writeRoot(node.dataset.croot);
    node.onchange = write;
    if (node.tagName !== 'SELECT') node.oninput = write;
    if (node.tagName === 'SELECT') {
      node.addEventListener('change', () => {
        const says = node.parentNode && node.parentNode.querySelector('.cardSays');
        const option = node.options[node.selectedIndex];
        if (says && option && option.title) says.textContent = option.title;
      });
    }
  });
  form.querySelectorAll('[data-ctoggle]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const box = node.closest('[data-cpath]');
      const text = box.querySelector('[data-cpart="text"]');
      const words = text.value.split(/[\s,]+/).filter(Boolean);
      const at = words.indexOf(node.dataset.ctoggle);
      if (at >= 0) words.splice(at, 1); else words.push(node.dataset.ctoggle);
      text.value = words.join(' ');
      node.classList.toggle('on', at < 0);
      writeRoot(box.dataset.croot);
    };
  });
  form.querySelectorAll('[data-cadd]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const root = node.dataset.crootAdd, path = JSON.parse(node.dataset.cadd);
      structural(root, target => addPart(target, root, path));
    };
  });
  form.querySelectorAll('[data-cdel]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const root = node.dataset.crootDel, path = JSON.parse(node.dataset.cdel);
      if (PICKING && PICKING.root === root) PICKING = null;
      structural(root, target => dropIn(target, path));
    };
  });
  form.querySelectorAll('[data-cpush]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      const root = node.dataset.crootPush, path = JSON.parse(node.dataset.cpush);
      const field = fieldAt(root, path);
      structural(root, target => {
        const list = getIn(target, path);
        const at = Array.isArray(list) ? list.length : 0;
        const item = field && field.control === 'list'
          ? fieldDefault(field.item, path.concat([at]), root) : {};
        if (Array.isArray(list)) list.push(item); else setIn(target, path, [item]);
      });
    };
  });
  form.querySelectorAll('[data-cmapadd]').forEach(node => {
    node.onchange = () => {
      if (!node.value) return;
      const root = node.dataset.crootMap, path = JSON.parse(node.dataset.cmapadd);
      const field = fieldAt(root, path);
      structural(root, target => setIn(target, path.concat([node.value]),
        field && field.control === 'map'
          ? fieldDefault(field.value, path.concat([node.value]), root) : null));
    };
  });
  form.querySelectorAll('[data-ceither]').forEach(node => {
    node.onchange = () => {
      const root = node.dataset.crootEither, path = JSON.parse(node.dataset.ceither);
      const field = fieldAt(root, path);
      if (!field || field.control !== 'either') return;
      const next = field.branches[Number(node.value)].field;
      structural(root, target => {
        const held = getIn(target, path);
        // One and several of the same thing convert into each other; anything else
        // starts again from the branch's default.
        let value = fieldDefault(next, path, root);
        if (next.control === 'list' && held && typeof held === 'object' && !Array.isArray(held)) value = [held];
        if (next.control === 'card' && Array.isArray(held) && held[0]) value = held[0];
        setIn(target, path, value);
      });
    };
  });
  form.querySelectorAll('[data-cchoice]').forEach(node => {
    node.onchange = () => switchChoice(node.dataset.crootChoice,
      JSON.parse(node.dataset.cchoice), Number(node.dataset.cgroup), Number(node.value));
  });
  form.querySelectorAll('[data-cpick]').forEach(node => {
    node.onclick = event => {
      event.preventDefault();
      const root = node.dataset.crootPick, path = JSON.parse(node.dataset.cpick);
      const next = Object.assign({ root, path, fromCard: true },
                                 pickNamed(root, path, node.dataset.clabel));
      const same = PICKING && PICKING.root === next.root
        && JSON.stringify(PICKING.path) === JSON.stringify(next.path);
      PICKING = same ? null : next;
      if (PICKING) { PLACING = null; PICK_LINE = ''; WILD_TOOL = null; }
      drawDesigner();
      drawVerdict();
      // The clicks go on the board, which may be a long way up the page (Q854).
      if (PICKING) boardToPick();
    };
  });
  form.querySelectorAll('[data-cshow]').forEach(node => {
    node.onclick = event => {
      event.stopPropagation();
      showDraft({ rule: node.dataset.cshow }, 'Playing games to find this rule at work…');
    };
  });
  // Choosing says what the rule does; a second click adds it. One step used to add the
  // rule before a word about it had been read, which is how *npcs* arrived beside a
  // strike eight actions before the validator refused the pair.
  const add = form.querySelector('#dAddRule');
  const go = form.querySelector('#dAddRuleGo');
  const says = form.querySelector('#dAddRuleSays');
  if (add && go) {
    add.onchange = () => {
      if (says) says.textContent = add.value ? ruleSays(add.value) : '';
      go.disabled = !add.value;
    };
    go.onclick = event => {
      event.preventDefault();
      const model = cardModel();
      const field = model && model.rules.find(f => f.key === add.value);
      if (!field) return;
      structural('rules', rules => {
        rules[field.key] = fieldDefault(field, [field.key], 'rules');
      });
    };
  }
}

// ---------------------------------------------------------------- a changed letter

// A piece's letter is its identity. Every placement, promotion, castling, capture list
// and chain names the piece by that letter, so changing it orphans all of them at once.
//
// Which fields hold a piece is the *field tree's* answer -- `web/cards.PIECE_PATHS` is
// what marks them and the schema is what it is derived from -- so the next rule that
// names a piece is carried here for nothing. Nothing below reads a letter to decide
// anything; it compares the one the draft held with the one that was typed.
function renamePiece(from, to) {
  if (!from || from === to) return;
  const model = cardModel();
  if (model) {
    renameInFields(model.rules, DRAFT.rules || {}, from, to);
    (DRAFT.pieces || []).forEach(piece => {
      renameInFields(model.piece, piece, from, to);
      (piece.moves || []).forEach(move => renameInFields(model.move, move, from, to));
    });
    ((DRAFT.setup || {}).placements || []).forEach(row =>
      renameInFields(model.placement, row, from, to));
  }
  // The two the hand-written form owns, which the tree therefore does not mark: the
  // placement row's piece, and the drawing filed under the piece's own letter.
  ((DRAFT.setup || {}).placements || []).forEach(row => {
    if (row.piece === from) row.piece = to;
  });
  const icons = (DRAFT.appearance || {}).icons;
  if (icons && from in icons) DRAFT.appearance.icons = renamedKey(icons, from, to);
}

// One key renamed, in place in the order the object already has, because the order of
// the keys is the order the form draws their rows in.
function renamedKey(object, from, to) {
  const out = {};
  Object.keys(object).forEach(key => { out[key === from ? to : key] = object[key]; });
  return out;
}

function renameInFields(fields, target, from, to) {
  if (!target || typeof target !== 'object') return;
  fields.forEach(field => renameInValue(field, target, field.key, from, to));
}

// One field of the tree, wherever the value it describes sits.
function renameInValue(field, holder, key, from, to) {
  const value = holder[key];
  if (value === undefined || value === null) return;
  if (field.control === 'piece') {
    if (value === from) holder[key] = to;
  } else if (field.control === 'pieces') {
    if (Array.isArray(value)) holder[key] = value.map(id => (id === from ? to : id));
  } else if (field.control === 'either') {
    renameInValue(field.branches[branchOf(field, value)].field, holder, key, from, to);
  } else if (field.control === 'card') {
    renameInFields(field.fields, value, from, to);
  } else if (field.control === 'list') {
    if (Array.isArray(value)) {
      value.forEach((_, at) => renameInValue(field.item, value, at, from, to));
    }
  } else if (field.control === 'map') {
    // A map is keyed by piece id, and its values can name a piece as well.
    Object.keys(value).forEach(id => renameInValue(field.value, value, id, from, to));
    if (from in value) holder[key] = renamedKey(value, from, to);
  }
}

// ---------------------------------------------------------------- squares on the board

function squareIndex(name, W) {
  const match = /^([a-z])(\d+)$/.exec(String(name));
  return match ? (Number(match[2]) - 1) * W + (match[1].charCodeAt(0) - 97) : -1;
}

// The squares the list being picked holds, as indices, for the opening board to tint.
function pickedSquares(W) {
  if (!PICKING) return null;
  const target = rootTarget(PICKING.root);
  const list = target ? getIn(target, PICKING.path) : null;
  return { moves: (Array.isArray(list) ? list : []).map(name => squareIndex(name, W))
    .filter(i => i >= 0), takes: [] };
}

// The list being picked, by name, for the pick note and the status line. With two wild
// pieces both their walks were *its walk* (the wolves walk, step 8,
// `docs/reference/designer-case-study-wolves.md`), so a field inside one entry of a rule's
// list is named as that entry's, the way its card heads it (*wild pieces 2's walk*), and
// the entry's piece comes with it where it names one (`piece`: *W*). Read off the draft's
// own entry; nothing here knows which rule it is.
function pickNamed(root, path, label) {
  const plain = { label: label || 'these squares', piece: '' };
  if (root !== 'rules' || path.length < 3 || typeof path[1] !== 'number') return plain;
  const key = String(path[0]);
  const owner = `${ruleTitle(key, key.replace(/_/g, ' '))} ${path[1] + 1}`;
  const said = plain.label;
  const named = /^its /.test(said) ? `${owner}'s ${said.slice(4)}` : `${said} of ${owner}`;
  const entry = getIn(DRAFT.rules || {}, path.slice(0, 2));
  return { label: named,
           piece: entry && typeof entry.piece === 'string' ? entry.piece : '' };
}

// What the pick note calls the list: its name and, where it has one, its piece's letter.
function pickCalled() {
  return PICKING.piece ? `${PICKING.label}, ${PICKING.piece}` : PICKING.label;
}

// A click on the opening board while a list is being picked: the square joins the list,
// or leaves it. A route is in click order, so a square joins at the end. The board is
// drawn again at once, from the draft (`pickedWalk`), and not only when the server's
// check comes back: two quick clicks used to aim the second at a picture a click behind
// (the wolves walk, step 7).
function pickSquare(square) {
  const preview = lastPreview();
  if (!PICKING || !preview || offBoard(preview, square)) return;
  const W = preview.width;
  const name = squareName(square % W, Math.floor(square / W));
  structural(PICKING.root, target => {
    const list = getIn(target, PICKING.path);
    const next = Array.isArray(list) ? list.slice() : [];
    const at = next.indexOf(name);
    if (at >= 0) next.splice(at, 1); else next.push(name);
    setIn(target, PICKING.path, next);
    benchSay(at >= 0 ? `Took ${name} out of ${PICKING.label}.`
      : `Added ${name} to ${PICKING.label}${next.length > 1 ? `, after ${next[next.length - 2]}` : ''}.`);
  });
  drawBoardSide();
}

// The wild piece's walk being picked, as the board draws it before the server's check of
// the click is back (Q854). The walk is the squares the draft's entry names, in the order
// it names them, which is what a route is (`schema.npc_routes`, unmirrored): the preview's
// walk for that entry is swapped for them, its pieces' rings and faint drawings move to the
// walk's starts, and a wild piece the preview still stands on a square that is no longer a
// start is not drawn there. The jumps kept are the server's own, where the step they name
// is still a step; a new jump is dashed when the answer comes. Once the answer agrees with
// the draft, its walk is drawn unchanged. Anything else being picked, the preview as it is.
function pickedWalk(preview) {
  const path = PICKING && PICKING.root === 'rules' ? PICKING.path : null;
  if (!preview || !path || path.length !== 3 || path[0] !== 'npcs' || path[2] !== 'route'
      || typeof path[1] !== 'number') return preview;
  const entry = path[1];
  const npc = ((DRAFT.rules || {}).npcs || [])[entry];
  if (!npc || typeof npc !== 'object') return preview;
  const W = preview.width, cells = W * preview.height;
  const absent = new Set(preview.absent || []);
  const index = name => squareIndex(name, W);
  const fits = at => at >= 0 && at < cells && !absent.has(at);
  const squares = [];
  (Array.isArray(npc.route) ? npc.route : []).map(index).forEach(at => {
    if (fits(at) && !squares.includes(at)) squares.push(at);
  });
  const walks = (preview.walks || []).slice();
  const at = walks.findIndex(walk => walk.entry === entry && !walk.mirrored);
  const held = at >= 0 ? walks[at] : null;
  if (held && JSON.stringify(held.squares || []) === JSON.stringify(squares)) return preview;
  const starts = squares.length ? [squares[0]] : [];
  (Array.isArray(npc.also_at) ? npc.also_at : []).map(index).forEach(start => {
    if (squares.includes(start) && !starts.includes(start)) starts.push(start);
  });
  const steps = new Set(squares.map((from, i) => `${from}>${squares[(i + 1) % squares.length]}`));
  const leaps = ((held && held.leaps) || []).filter(leap => steps.has(`${leap.from}>${leap.to}`));
  const type = (DRAFT.pieces || []).findIndex(piece => piece.id === npc.piece);
  const walk = {
    name: held ? held.name : `${ruleTitle('npcs', 'wild pieces')} ${entry + 1}`,
    entry, mirrored: false, piece: typeof npc.piece === 'string' ? npc.piece : null,
    type: type >= 0 && type < (preview.nTypes || 0) ? type : (held ? held.type : null),
    squares, starts, leaps,
    says: held && leaps.length === (held.leaps || []).length ? held.says : null,
  };
  if (held) walks[at] = walk;
  else {
    const after = walks.findIndex(other => other.entry > entry);
    walks.splice(after < 0 ? walks.length : after, 0, walk);
  }
  // The wild piece the preview stands on an old start, taken off this drawing only: the
  // faint one `walksHtml` draws on each new start stands in for it until the answer.
  const stride = preview.sideStride || preview.nTypes || 0;
  const board = (preview.board || []).slice();
  ((held && held.starts) || []).forEach(square => {
    if (!starts.includes(square) && board[square] > 2 * stride) board[square] = 0;
  });
  return Object.assign({}, preview, { walks, board });
}

function pickNoteHtml() {
  if (!PICKING) return '';
  return `<div class="small pickNote" data-picknote style="margin-top:6px"><span class="muted">You
    are picking squares for</span> <b>${esc(pickCalled())}</b><span class="muted">. Click squares
    on the opening board, and click one again to take it back. Squares are written for the
    first player.</span> <button class="small" data-pickdone>done picking</button></div>${
    pickLineHtml()}`;
}

// *done picking* under the board: the same as the card's own button, and a pick begun on
// the card brings the card back into view where the board had taken its place (a phone,
// Q854). One begun beside the board (*+ wild piece*) stays there.
function donePicking() {
  if (!PICKING) return;
  const was = PICKING;
  PICKING = null;
  PICK_LINE = '';
  drawDesigner();
  drawVerdict();
  if (!was.fromCard) return;
  const button = [...document.querySelectorAll('[data-cpick]')].find(node =>
    node.dataset.crootPick === was.root && node.dataset.cpick === JSON.stringify(was.path));
  if (!button || typeof button.getBoundingClientRect !== 'function') return;
  const box = button.getBoundingClientRect();
  if (box.top < 0 || box.bottom > window.innerHeight - footbarRoom()) {
    button.scrollIntoView({ block: 'center' });
  }
}
