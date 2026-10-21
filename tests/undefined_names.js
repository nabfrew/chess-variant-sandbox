// Every name a page's script reads or writes that nothing declares: `no-undef`, without npm.
//
// `tests/page_names.py` runs this as `node --expose-internals tests/undefined_names.js`, with
// a JSON object `{label: source}` on stdin, and reads back `{label: [findings]}`: one
// `{name, line}` per undefined name and the line it is first used on, or `{error}` where a
// script does not parse. The parser is the acorn node carries for its own REPL
// (`process.versions.acorn`), so there is no package to install; the walk is below.
//
// A script is checked as a classic script, the way the browser runs it: one global scope,
// function declarations hoisted, `var` hoisted to its function, `let`, `const` and `class`
// to their block. Order does not matter -- this finds names that are declared nowhere, not
// names used before their line. A name is defined if some enclosing scope declares it, the
// language has it (read off a fresh `vm` context, so it follows the node running it), or
// it is on `BROWSER` below. `typeof name` is allowed to name anything, as `no-undef` allows.
'use strict';

const vm = require('vm');
const acorn = require('internal/deps/acorn/acorn/dist/acorn');

// What the browser, and the vendored libraries the page loads before its script, put in
// the global scope, as far as a page uses them. A new one is added here when a page first
// reads it; the test that fails names it.
const BROWSER = [
  'window', 'document', 'navigator', 'location', 'history', 'screen', 'console',
  'localStorage', 'sessionStorage', 'fetch', 'Headers', 'Request', 'Response',
  'AbortController', 'AbortSignal', 'URL', 'URLSearchParams', 'FormData', 'Blob', 'File',
  'FileReader', 'TextEncoder', 'TextDecoder', 'Worker', 'Event', 'CustomEvent',
  'KeyboardEvent', 'MouseEvent', 'Node', 'Element', 'HTMLElement', 'HTMLInputElement',
  'DOMParser', 'XMLSerializer', 'MutationObserver', 'ResizeObserver', 'CSSStyleSheet',
  'IntersectionObserver', 'Image', 'performance', 'crypto', 'structuredClone',
  'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'queueMicrotask',
  'requestAnimationFrame', 'cancelAnimationFrame', 'getComputedStyle', 'matchMedia',
  'alert', 'confirm', 'prompt', 'open', 'close', 'print', 'scrollTo', 'scrollBy',
  'atob', 'btoa', 'CSS', 'innerWidth', 'innerHeight', 'devicePixelRatio',
  'addEventListener', 'removeEventListener', 'dispatchEvent', 'self', 'origin',
  'cytoscape',
];

const DEFINED = new Set([
  ...vm.runInNewContext('Object.getOwnPropertyNames(globalThis)'),
  ...BROWSER,
  'arguments',
]);

class Scope {
  constructor(parent, isFunction) {
    this.parent = parent;
    this.isFunction = isFunction;
    this.names = new Set();
  }
  hoisting() {
    let scope = this;
    while (!scope.isFunction) scope = scope.parent;
    return scope;
  }
  has(name) {
    for (let scope = this; scope; scope = scope.parent) if (scope.names.has(name)) return true;
    return false;
  }
}

function check(source) {
  let program;
  try {
    program = acorn.parse(source, { ecmaVersion: 'latest', sourceType: 'script', locations: true });
  } catch (error) {
    return { error: String(error.message) };
  }
  // Every reference and the scope it sits in; resolved once the whole script is walked,
  // so a declaration further down (hoisted, or simply later) counts.
  const references = [];
  const refer = (node, scope) => references.push({ node, scope });

  // The names a binding pattern declares, and the expressions inside it (defaults,
  // computed keys), which are ordinary references in `scope`.
  function declare(pattern, into, scope) {
    switch (pattern.type) {
      case 'Identifier': into.names.add(pattern.name); break;
      case 'ObjectPattern':
        for (const property of pattern.properties) {
          if (property.type === 'RestElement') declare(property.argument, into, scope);
          else {
            if (property.computed) walk(property.key, scope);
            declare(property.value, into, scope);
          }
        }
        break;
      case 'ArrayPattern':
        for (const element of pattern.elements) if (element) declare(element, into, scope);
        break;
      case 'RestElement': declare(pattern.argument, into, scope); break;
      case 'AssignmentPattern':
        declare(pattern.left, into, scope);
        walk(pattern.right, scope);
        break;
      default: throw new Error('unexpected binding ' + pattern.type);
    }
  }

  // An assignment target: every plain name in it is a reference (a write to an undeclared
  // name is as much a slip as a read), and a member expression is walked as one.
  function target(pattern, scope) {
    switch (pattern.type) {
      case 'Identifier': refer(pattern, scope); break;
      case 'ObjectPattern':
        for (const property of pattern.properties) {
          if (property.type === 'RestElement') target(property.argument, scope);
          else {
            if (property.computed) walk(property.key, scope);
            target(property.value, scope);
          }
        }
        break;
      case 'ArrayPattern':
        for (const element of pattern.elements) if (element) target(element, scope);
        break;
      case 'RestElement': target(pattern.argument, scope); break;
      case 'AssignmentPattern': target(pattern.left, scope); walk(pattern.right, scope); break;
      default: walk(pattern, scope);
    }
  }

  // Declarations are entered into their scope before the body is walked; which scope
  // depends on the kind, and in a classic script a function declared in a block is also
  // hoisted to its function (Annex B), so it goes into both.
  function hoist(statements, block) {
    for (const statement of statements) hoistOne(statement, block);
  }
  function hoistOne(node, block) {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
      case 'FunctionDeclaration':
        block.names.add(node.id.name);
        block.hoisting().names.add(node.id.name);
        return;
      case 'ClassDeclaration': block.names.add(node.id.name); return;
      case 'VariableDeclaration':
        if (node.kind !== 'var') {
          for (const d of node.declarations) declareNames(d.id, block);
          return;
        }
        for (const d of node.declarations) declareNames(d.id, block.hoisting());
        return;
      default:
    }
    // `var` inside nested statements (not functions) hoists out of them.
    for (const key of Object.keys(node)) {
      if (key === 'type' || key === 'loc') continue;
      const value = node[key];
      for (const child of Array.isArray(value) ? value : [value]) {
        if (child && typeof child.type === 'string') hoistVar(child, block.hoisting());
      }
    }
  }
  function hoistVar(node, into) {
    if (!node || typeof node.type !== 'string') return;
    if (/Function/.test(node.type) || node.type === 'ClassDeclaration'
      || node.type === 'ClassExpression') return;
    if (node.type === 'VariableDeclaration' && node.kind === 'var') {
      for (const d of node.declarations) declareNames(d.id, into);
    }
    for (const key of Object.keys(node)) {
      if (key === 'type' || key === 'loc') continue;
      const value = node[key];
      for (const child of Array.isArray(value) ? value : [value]) {
        if (child && typeof child.type === 'string') hoistVar(child, into);
      }
    }
  }
  // Only the names; defaults and computed keys are walked where the declaration runs.
  function declareNames(pattern, into) {
    switch (pattern.type) {
      case 'Identifier': into.names.add(pattern.name); break;
      case 'ObjectPattern':
        for (const p of pattern.properties) declareNames(p.type === 'RestElement' ? p.argument : p.value, into);
        break;
      case 'ArrayPattern': for (const e of pattern.elements) if (e) declareNames(e, into); break;
      case 'RestElement': declareNames(pattern.argument, into); break;
      case 'AssignmentPattern': declareNames(pattern.left, into); break;
      default:
    }
  }

  function walkFunction(node, scope) {
    const inner = new Scope(scope, true);
    if (node.type === 'FunctionExpression' && node.id) inner.names.add(node.id.name);
    for (const param of node.params) declare(param, inner, inner);
    if (node.body.type === 'BlockStatement') {
      hoist(node.body.body, inner);
      for (const statement of node.body.body) walk(statement, inner);
    } else {
      walk(node.body, inner);
    }
  }

  function walkBlock(statements, scope) {
    const block = new Scope(scope, false);
    hoist(statements, block);
    for (const statement of statements) walk(statement, block);
  }

  function walk(node, scope) {
    if (!node || typeof node.type !== 'string') return;
    switch (node.type) {
      case 'Identifier': refer(node, scope); return;
      case 'FunctionDeclaration': case 'FunctionExpression': case 'ArrowFunctionExpression':
        walkFunction(node, scope); return;
      case 'ClassDeclaration': case 'ClassExpression': {
        const inner = new Scope(scope, false);
        if (node.id) inner.names.add(node.id.name);
        walk(node.superClass, inner);
        for (const member of node.body.body) {
          if (member.computed) walk(member.key, inner);
          if (member.type === 'StaticBlock') walkBlock(member.body, new Scope(inner, true));
          else walk(member.value, new Scope(inner, true));
        }
        return;
      }
      case 'VariableDeclaration':
        for (const d of node.declarations) {
          declare(d.id, new Scope(null, true), scope);
          walk(d.init, scope);
        }
        return;
      case 'BlockStatement': walkBlock(node.body, scope); return;
      case 'StaticBlock': walkBlock(node.body, scope); return;
      case 'ForStatement': case 'ForInStatement': case 'ForOfStatement': {
        const loop = new Scope(scope, false);
        const head = node.type === 'ForStatement' ? node.init : node.left;
        if (head && head.type === 'VariableDeclaration') hoistOne(head, loop);
        else if (head && node.type !== 'ForStatement') target(head, loop);
        else walk(head, loop);
        if (head && head.type === 'VariableDeclaration') walk(head, loop);
        walk(node.test, loop);
        walk(node.update, loop);
        walk(node.right, loop);
        walk(node.body, loop);
        return;
      }
      case 'SwitchStatement': {
        walk(node.discriminant, scope);
        const block = new Scope(scope, false);
        const all = node.cases.flatMap(c => c.consequent);
        hoist(all, block);
        for (const c of node.cases) {
          walk(c.test, block);
          for (const statement of c.consequent) walk(statement, block);
        }
        return;
      }
      case 'CatchClause': {
        const inner = new Scope(scope, false);
        if (node.param) declare(node.param, inner, inner);
        walk(node.body, inner);
        return;
      }
      case 'AssignmentExpression': target(node.left, scope); walk(node.right, scope); return;
      case 'UpdateExpression': target(node.argument, scope); return;
      case 'MemberExpression':
        walk(node.object, scope);
        if (node.computed) walk(node.property, scope);
        return;
      case 'Property': case 'PropertyDefinition': case 'MethodDefinition':
        if (node.computed) walk(node.key, scope);
        walk(node.value, scope);
        return;
      case 'UnaryExpression':
        if (node.operator === 'typeof' && node.argument.type === 'Identifier') return;
        walk(node.argument, scope);
        return;
      case 'LabeledStatement': walk(node.body, scope); return;
      case 'BreakStatement': case 'ContinueStatement': case 'MetaProperty':
      case 'PrivateIdentifier': case 'Literal': case 'ThisExpression': case 'Super':
        return;
      default:
    }
    for (const key of Object.keys(node)) {
      if (key === 'type' || key === 'loc') continue;
      const value = node[key];
      for (const child of Array.isArray(value) ? value : [value]) walk(child, scope);
    }
  }

  const global = new Scope(null, true);
  hoist(program.body, global);
  for (const statement of program.body) walk(statement, global);

  const found = new Map();
  for (const { node, scope } of references) {
    if (scope.has(node.name) || DEFINED.has(node.name) || found.has(node.name)) continue;
    found.set(node.name, node.loc.start.line);
  }
  return [...found].map(([name, line]) => ({ name, line }));
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  const scripts = JSON.parse(input);
  const answer = {};
  for (const [label, source] of Object.entries(scripts)) answer[label] = check(source);
  process.stdout.write(JSON.stringify(answer));
});
