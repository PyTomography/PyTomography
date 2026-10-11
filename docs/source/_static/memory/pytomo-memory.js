/* PyTomography's memory model in the browser: evaluates model.json (PyTomography's estimate_memory() and
 * plan_memory() as data) with no dependencies and no build step. Load it with a <script> tag (it defines
 * window.PyTomoMemory) or require() it in Node (module.exports). docs/tools/memory_model.py evaluates the same file in
 * Python, and docs/tools/check_memory_model.py proves the two agree with each other and with the library.
 *
 *   PyTomoMemory.load(modelJson)                        // or PyTomoMemory.loadUrl(url) -> Promise
 *   PyTomoMemory.models()                               // [{id, title, modality, params: [...], settings: [{name, label}]}]
 *   PyTomoMemory.estimate(id, params, settings)         // {ram_gb, gpu_gb, parts: [{name, scope, ram_gb, gpu_gb}], settings}
 *   PyTomoMemory.plan(id, params, ram_gb, gpu_gb, fixed) // {settings, estimate, fits, options: [{settings, ram_gb, gpu_gb, fits}]}
 *   PyTomoMemory.code(id, params, settings)             // a Python snippet with these settings that prints estimate_memory()
 *
 * The expression language is documented at the top of model.json; nothing outside it is evaluated (no eval).
 */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module && module.exports) { module.exports = api; }
  if (typeof window !== 'undefined') { window.PyTomoMemory = api; }
  else if (root && typeof root === 'object') { root.PyTomoMemory = api; }
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  // ------------------------------------------------------------------------------------------ expression language
  var TWO = ['**', '<=', '>=', '==', '!=', '&&', '||'];
  var ONE = '+-*/%<>!?:(),';

  function isDigit(c) { return c >= '0' && c <= '9'; }
  function isAlpha(c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_'; }

  function tokens(text) {
    var out = [], i = 0, n = text.length;
    while (i < n) {
      var c = text[i];
      if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; }
      else if (isDigit(c) || (c === '.' && i + 1 < n && isDigit(text[i + 1]))) {
        var j = i;
        while (j < n && (isDigit(text[j]) || text[j] === '.')) { j++; }
        if (j < n && (text[j] === 'e' || text[j] === 'E')) {
          var k = j + 1;
          if (k < n && (text[k] === '+' || text[k] === '-')) { k++; }
          if (k < n && isDigit(text[k])) { j = k; while (j < n && isDigit(text[j])) { j++; } }
        }
        out.push(['num', parseFloat(text.slice(i, j))]);
        i = j;
      } else if (isAlpha(c)) {
        var m = i;
        while (m < n && (isAlpha(text[m]) || isDigit(text[m]))) { m++; }
        out.push(['name', text.slice(i, m)]);
        i = m;
      } else if (c === '"' || c === "'") {
        var e = text.indexOf(c, i + 1);
        if (e < 0) { throw new Error('unterminated string in ' + text); }
        out.push(['str', text.slice(i + 1, e)]);
        i = e + 1;
      } else if (TWO.indexOf(text.slice(i, i + 2)) >= 0) { out.push(['op', text.slice(i, i + 2)]); i += 2; }
      else if (ONE.indexOf(c) >= 0) { out.push(['op', c]); i++; }
      else { throw new Error('unexpected character ' + c + ' in ' + text); }
    }
    return out;
  }

  function Parser(text) { this.text = text; this.toks = tokens(text); this.i = 0; }
  Parser.prototype.peek = function (value) {
    if (this.i < this.toks.length) {
      var t = this.toks[this.i];
      if (value === undefined || (t[0] === 'op' && t[1] === value)) { return t; }
    }
    return null;
  };
  Parser.prototype.take = function (value) {
    if (!this.peek(value)) { throw new Error('expected ' + value + ' in ' + this.text); }
    this.i++;
  };
  Parser.prototype.parse = function () {
    var tree = this.ternary();
    if (this.i !== this.toks.length) { throw new Error('unexpected ' + this.toks[this.i][1] + ' in ' + this.text); }
    return tree;
  };
  Parser.prototype.ternary = function () {
    var cond = this.orr();
    if (this.peek('?')) {
      this.i++;
      var a = this.ternary();
      this.take(':');
      var b = this.ternary();
      return ['if', cond, a, b];
    }
    return cond;
  };
  Parser.prototype.orr = function () {
    var a = this.andd();
    while (this.peek('||')) { this.i++; a = ['or', a, this.andd()]; }
    return a;
  };
  Parser.prototype.andd = function () {
    var a = this.nott();
    while (this.peek('&&')) { this.i++; a = ['and', a, this.nott()]; }
    return a;
  };
  Parser.prototype.nott = function () {
    if (this.peek('!')) { this.i++; return ['not', this.nott()]; }
    return this.cmp();
  };
  Parser.prototype.cmp = function () {
    var a = this.add();
    var ops = ['<=', '>=', '==', '!=', '<', '>'];
    for (var k = 0; k < ops.length; k++) {
      if (this.peek(ops[k])) { this.i++; return ['bin', ops[k], a, this.add()]; }
    }
    return a;
  };
  Parser.prototype.add = function () {
    var a = this.mul();
    while (this.peek('+') || this.peek('-')) { var op = this.toks[this.i][1]; this.i++; a = ['bin', op, a, this.mul()]; }
    return a;
  };
  Parser.prototype.mul = function () {
    var a = this.unary();
    while (this.peek('*') || this.peek('/') || this.peek('%')) { var op = this.toks[this.i][1]; this.i++; a = ['bin', op, a, this.unary()]; }
    return a;
  };
  Parser.prototype.unary = function () {
    if (this.peek('-')) { this.i++; return ['neg', this.unary()]; }
    if (this.peek('+')) { this.i++; return this.unary(); }
    return this.power();
  };
  Parser.prototype.power = function () {
    var a = this.atom();
    if (this.peek('**')) { this.i++; return ['bin', '**', a, this.unary()]; }
    return a;
  };
  Parser.prototype.atom = function () {
    var t = this.peek();
    if (!t) { throw new Error('unexpected end of ' + this.text); }
    this.i++;
    if (t[0] === 'num') { return ['num', t[1]]; }
    if (t[0] === 'str') { return ['str', t[1]]; }
    if (t[0] === 'name') {
      if (this.peek('(')) {
        this.i++;
        var args = [];
        if (!this.peek(')')) {
          args.push(this.ternary());
          while (this.peek(',')) { this.i++; args.push(this.ternary()); }
        }
        this.take(')');
        if (!Object.prototype.hasOwnProperty.call(FUNCTIONS, t[1])) { throw new Error('unknown function ' + t[1] + ' in ' + this.text); }
        return ['call', t[1], args];
      }
      return ['name', t[1]];
    }
    if (t[1] === '(') { var tree = this.ternary(); this.take(')'); return tree; }
    throw new Error('unexpected ' + t[1] + ' in ' + this.text);
  };

  function num(x) {
    if (x === null || x === undefined || typeof x === 'string') { throw new Error('a number was expected, got ' + x); }
    return +x;
  }

  // x rounded to n significant digits as Python's float('%.{n}g' % x): correctly rounded, ties to even
  function sig(x, n) {
    n = Math.max(1, Math.round(n));
    if (x === 0 || !isFinite(x)) { return x; }
    var rounded = Number(x.toPrecision(n));            // correctly rounded, but ties away from zero
    var full = Math.abs(x).toExponential(99);           // the exact value's first 100 digits
    var mant = full.split('e')[0].replace('.', ''), expo = full.split('e')[1];
    if (mant[n] === '5' && /^0*$/.test(mant.slice(n + 1)) && parseInt(mant[n - 1], 10) % 2 === 0) {
      // exactly halfway, after an even digit: Python keeps the even digit
      return (x < 0 ? -1 : 1) * Number(mant[0] + (n > 1 ? '.' + mant.slice(1, n) : '') + 'e' + expo);
    }
    return rounded;
  }

  var FUNCTIONS = {
    min: function () { var m = Infinity; for (var i = 0; i < arguments.length; i++) { m = Math.min(m, num(arguments[i])); } return m; },
    max: function () { var m = -Infinity; for (var i = 0; i < arguments.length; i++) { m = Math.max(m, num(arguments[i])); } return m; },
    ceil: function (x) { return Math.ceil(num(x)); },
    floor: function (x) { return Math.floor(num(x)); },
    round: function (x) { x = num(x); return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x); },
    abs: function (x) { return Math.abs(num(x)); },
    sqrt: function (x) { return Math.sqrt(num(x)); },
    log2: function (x) { return Math.log2(num(x)); },
    log10: function (x) { return Math.log10(num(x)); },
    sin: function (x) { return Math.sin(num(x)); },
    cos: function (x) { return Math.cos(num(x)); },
    asin: function (x) { return Math.asin(num(x)); },
    sig: function (x, n) { return sig(num(x), num(n)); }
  };
  var CONSTANTS = { 'true': 1, 'false': 0, 'null': null, 'pi': Math.PI };

  function text(x) { return x === null || x === undefined ? 'null' : String(x); }
  function truthy(x) { return x !== null && x !== undefined && x !== 0 && x !== ''; }
  function div(a, b) { if (b === 0) { return a === 0 ? NaN : (a > 0 ? Infinity : -Infinity); } return a / b; }

  function evalTree(t, env) {
    switch (t[0]) {
      case 'num': case 'str': return t[1];
      case 'name':
        if (Object.prototype.hasOwnProperty.call(env, t[1])) { return env[t[1]]; }
        if (Object.prototype.hasOwnProperty.call(CONSTANTS, t[1])) { return CONSTANTS[t[1]]; }
        throw new Error('unknown name ' + t[1]);
      case 'neg': return -num(evalTree(t[1], env));
      case 'not': return truthy(evalTree(t[1], env)) ? 0 : 1;
      case 'and': return (truthy(evalTree(t[1], env)) && truthy(evalTree(t[2], env))) ? 1 : 0;
      case 'or': return (truthy(evalTree(t[1], env)) || truthy(evalTree(t[2], env))) ? 1 : 0;
      case 'if': return evalTree(truthy(evalTree(t[1], env)) ? t[2] : t[3], env);
      case 'call': return FUNCTIONS[t[1]].apply(null, t[2].map(function (a) { return evalTree(a, env); }));
    }
    var op = t[1], a = evalTree(t[2], env), b = evalTree(t[3], env);
    if (op === '==') { return a === b ? 1 : 0; }
    if (op === '!=') { return a === b ? 0 : 1; }
    if (op === '+' && (typeof a === 'string' || typeof b === 'string')) { return text(a) + text(b); }
    a = num(a); b = num(b);
    switch (op) {
      case '+': return a + b;
      case '-': return a - b;
      case '*': return a * b;
      case '/': return div(a, b);
      case '%': return b !== 0 ? a - b * Math.floor(div(a, b)) : NaN;
      case '**': return Math.pow(a, b);
      case '<': return a < b ? 1 : 0;
      case '<=': return a <= b ? 1 : 0;
      case '>': return a > b ? 1 : 0;
      case '>=': return a >= b ? 1 : 0;
    }
    throw new Error('unknown operator ' + op);
  }

  var cache = {};
  function evaluate(expression, env) {
    if (expression === null || expression === undefined) { return null; }
    if (typeof expression === 'boolean') { return expression ? 1 : 0; }
    if (typeof expression === 'number') { return expression; }
    var tree = cache[expression];
    if (!tree) { tree = cache[expression] = new Parser(expression).parse(); }
    return evalTree(tree, env);
  }

  function value(x) { return typeof x === 'boolean' ? (x ? 1 : 0) : x; }
  function out(x, kind) {
    if (x === null || x === undefined || typeof x === 'string') { return x === undefined ? null : x; }
    if (kind === 'bool') { return !!x; }
    return x;
  }

  // ------------------------------------------------------------------------------------------------- the model
  var data = null, variables = {}, byId = {};

  function load(modelJson) {
    data = typeof modelJson === 'string' ? JSON.parse(modelJson) : modelJson;
    variables = {}; byId = {};
    data.variables.forEach(function (v) { variables[v.name] = v; });
    data.models.forEach(function (m) { byId[m.id] = m; });
    return api;
  }

  function loadUrl(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) { throw new Error('could not load ' + url + ': ' + r.status); }
      return r.json();
    }).then(load);
  }

  function model(id) {
    if (!data) { throw new Error('PyTomoMemory: load(model.json) first'); }
    var m = byId[id];
    if (!m) { throw new Error('PyTomoMemory: no model ' + id); }
    return m;
  }

  function paramsOf(m) {
    var defaults = m.defaults || {};
    return m.params.map(function (name) {
      var p = {}, v = variables[name];
      Object.keys(v).forEach(function (k) { p[k] = v[k]; });
      if (Object.prototype.hasOwnProperty.call(defaults, name)) { p['default'] = defaults[name]; }
      return p;
    });
  }

  function models() {
    var keys = ['name', 'label', 'unit', 'default', 'min', 'max', 'step', 'options', 'type', 'help'];
    return data.models.map(function (m) {
      return {
        id: m.id, title: m.title, modality: m.modality,
        params: paramsOf(m).map(function (p) { var q = {}; keys.forEach(function (k) { if (k in p) { q[k] = p[k]; } }); return q; }),
        settings: (m.settings || []).map(function (s) { return {name: s.name, label: s.label || s.name}; })
      };
    });
  }

  function eachDerived(group, fn) {
    if (!group) { return; }
    if (Array.isArray(group)) { group.forEach(function (pair) { fn(pair[0], pair[1]); }); }
    else { Object.keys(group).forEach(function (k) { fn(k, group[k]); }); }
  }

  function makeEnv(m, params, settings) {
    var env = {};
    paramsOf(m).forEach(function (p) { env[p.name] = value(p['default'] === undefined ? null : p['default']); });
    Object.keys(params || {}).forEach(function (k) { env[k] = value(params[k]); });
    Object.keys(settings || {}).forEach(function (k) { env[k] = value(settings[k]); });
    var groups = [data.derived].concat((m.use || []).map(function (n) { return data.derived_sets[n]; })).concat([m.derived]);
    groups.forEach(function (g) { eachDerived(g, function (k, expr) { env[k] = evaluate(expr, env); }); });
    return env;
  }

  function activeSettings(m, env) {
    return (m.settings || []).filter(function (s) { return !('when' in s) || truthy(evaluate(s.when, env)); });
  }

  function forced(s, env) { return 'force' in s ? evaluate(s.force, env) : null; }

  function resolve(m, params, settings) {
    var env0 = makeEnv(m, params);
    var given = {};
    Object.keys(settings || {}).forEach(function (k) { given[k] = settings[k]; });
    activeSettings(m, env0).forEach(function (s) { var f = forced(s, env0); if (f !== null) { given[s.name] = f; } });
    var env = makeEnv(m, params, given);
    var used = {};
    activeSettings(m, env).forEach(function (s) { used[s.name] = out(evaluate(s.report || s.name, env), s.type); });
    return {env: env, used: used};
  }

  function expand(parts) {
    var res = [];
    parts.forEach(function (p) {
      if ('include' in p) { res = res.concat(expand(data.part_sets[p.include])); } else { res.push(p); }
    });
    return res;
  }

  function fmt(x) { return (typeof x === 'number' && Number.isInteger(x)) ? String(x) : String(x); }

  function name(template, env) {
    return template.replace(/\{([^}]*)\}/g, function (_, expr) { return fmt(out(evaluate(expr, env))); });
  }

  function addPart(p, env, res) {
    if ('when' in p && !truthy(evaluate(p.when, env))) { return; }
    var ram, gpu;
    if ('bytes' in p) {
      var b = num(evaluate(p.bytes, env));
      var onGpu = truthy(evaluate('on' in p ? p.on : 0, env));
      ram = onGpu ? 0 : b; gpu = onGpu ? b : 0;
    } else {
      ram = num(evaluate('ram' in p ? p.ram : 0, env));
      gpu = num(evaluate('gpu' in p ? p.gpu : 0, env));
    }
    res.push([name(p.name, env), p.scope || 'held', ram, gpu]);
  }

  function ownParts(m, env) {
    var res = [];
    expand(m.parts).forEach(function (p) {
      if ('repeat' in p) {
        var r = p.repeat, lo = Math.trunc(num(evaluate(r.from, env))), hi = Math.trunc(num(evaluate(r.to, env)));
        var local = Object.create(null);
        Object.keys(env).forEach(function (k) { local[k] = env[k]; });
        for (var k = lo; k <= hi; k++) { local[r['var']] = k; addPart(p, local, res); }
      } else {
        addPart(p, env, res);
      }
    });
    return res;
  }

  function sum(parts, k) { var s = 0; for (var i = 0; i < parts.length; i++) { s += parts[i][k]; } return s; }

  function withOverheads(m, env, parts) {
    var o = data.overheads, res = parts.slice();
    res.push([o.fixed.name, 'overhead', num(evaluate(o.fixed.bytes, env)), 0]);
    var gpuArrays = sum(parts, 3);
    if (gpuArrays > 0) { res.push([o.cuda_cache.name, 'overhead', 0, o.cuda_cache.bytes + o.cuda_cache.fraction * gpuArrays]); }
    if (truthy(evaluate(o.gpu_counts_as_ram, env))) {
      var arrays = 0;
      for (var i = 0; i < parts.length; i++) { arrays += parts[i][2] + parts[i][3]; }
      var fraction = 'allocator_fraction' in m ? m.allocator_fraction : o.allocator.fraction;
      if (arrays > 0) { res.push([o.allocator.name, 'overhead', fraction * arrays, 0]); }
    }
    return res;
  }

  function estimate(id, params, settings) {
    var m = model(id);
    var r = resolve(m, params, settings);
    var parts = withOverheads(m, r.env, ownParts(m, r.env));
    var gpu = sum(parts, 3), ram = sum(parts, 2);
    if (truthy(evaluate(data.overheads.gpu_counts_as_ram, r.env))) { ram += gpu; }
    return {
      model: id, ram_gb: ram / 1e9, gpu_gb: gpu / 1e9, settings: r.used,
      parts: parts.map(function (p) { return {name: p[0], scope: p[1], ram_gb: p[2] / 1e9, gpu_gb: p[3] / 1e9}; })
    };
  }

  function candidateList(s, env) {
    var f = forced(s, env);
    if (f !== null) { return [f]; }
    var spec = s.candidates, values = [];
    spec.items.forEach(function (item) {
      if ('value' in item) {
        if (!('if' in item) || truthy(evaluate(item['if'], env))) { values.push(evaluate(item.value, env)); }
        return;
      }
      var seq = item.seq;
      var lo = Math.trunc(num(evaluate(seq.from, env))), hi = Math.trunc(num(evaluate(seq.to, env)));
      var local = Object.create(null);
      Object.keys(env).forEach(function (k) { local[k] = env[k]; });
      for (var k = lo; k <= hi; k++) {
        local[seq['var']] = k;
        if ('if' in seq) {
          local.value = evaluate(seq.value, local);
          if (!truthy(evaluate(seq['if'], local))) { continue; }
        }
        values.push(evaluate(seq.value, local));
      }
    });
    if (spec.unique) {
      var u = [];
      values.forEach(function (v) { if (u.indexOf(v) < 0) { u.push(v); } });
      values = u;
    }
    if (spec.sort === 'asc') { values = values.slice().sort(function (a, b) { return a - b; }); }
    return values;
  }

  function candidates(id, params) {
    var m = model(id), env = makeEnv(m, params), res = {};
    activeSettings(m, env).forEach(function (s) { res[s.name] = candidateList(s, env).map(function (v) { return out(v, s.type); }); });
    return res;
  }

  function plan(id, params, ramGb, gpuGb, fixed) {
    var m = model(id), env = makeEnv(m, params);
    ramGb = (ramGb === null || ramGb === undefined) ? Infinity : ramGb;
    gpuGb = (gpuGb === null || gpuGb === undefined) ? Infinity : gpuGb;
    var keep = {};
    Object.keys(fixed || {}).forEach(function (k) { if (fixed[k] !== null && fixed[k] !== undefined) { keep[k] = fixed[k]; } });
    var active = activeSettings(m, env);
    var names = active.map(function (s) { return s.name; });
    var types = {};
    active.forEach(function (s) { types[s.name] = s.type; });
    Object.keys(keep).forEach(function (k) {
      if (names.indexOf(k) < 0) { throw new Error(id + ' has no setting ' + k + ': its settings are ' + (names.join(', ') || 'none')); }
    });
    var kept = [];
    var lists = active.map(function (s) {
      var f = forced(s, env);
      if (f !== null) { kept.push(s.name); return [f]; }
      if (s.name in keep) { kept.push(s.name); return [value(keep[s.name])]; }
      var c = candidateList(s, env);
      return c.length ? c : [env[s.name]];
    });
    function report(trial) { var r = {}; names.forEach(function (n) { r[n] = out(trial[n], types[n]); }); return r; }
    var options = [], chosen = null, closest = null, least = Infinity;
    var idx = lists.map(function () { return 0; });
    var total = lists.reduce(function (a, l) { return a * l.length; }, 1);
    for (var count = 0; count < total; count++) {
      var trial = {};
      names.forEach(function (n, i) { trial[n] = lists[i][idx[i]]; });
      var e = estimate(id, params, trial);
      var excess = Math.max(0, e.ram_gb - ramGb) + Math.max(0, e.gpu_gb - gpuGb);
      var fits = excess <= 1e-9;
      options.push({settings: report(trial), ram_gb: e.ram_gb, gpu_gb: e.gpu_gb, fits: fits});
      if (fits) { chosen = trial; break; }
      if (excess < least) { closest = trial; least = excess; }
      for (var j = lists.length - 1; j >= 0; j--) {       // the last setting is the inner loop
        idx[j]++;
        if (idx[j] < lists[j].length) { break; }
        idx[j] = 0;
      }
    }
    var settings = chosen !== null ? chosen : closest;
    return {
      model: id, settings: report(settings), estimate: estimate(id, params, settings), fits: chosen !== null,
      options: options, ram_gb: ramGb, gpu_gb: gpuGb, fixed: names.filter(function (n) { return kept.indexOf(n) >= 0; })
    };
  }

  function literal(v, kind) {
    if (v === null || v === undefined) { return 'None'; }
    if (typeof v === 'boolean' || kind === 'bool') { return v ? 'True' : 'False'; }
    if (typeof v === 'string') { return "'" + v + "'"; }
    if (Number.isInteger(v) && Math.abs(v) < 9007199254740992) { return String(v); }
    var s = String(v);
    return /[.eE]/.test(s) || !isFinite(v) ? s : s + '.0';
  }

  function code(id, params, settings) {
    var m = model(id), r = resolve(m, params, settings), kinds = {};
    Object.keys(variables).forEach(function (k) { kinds[k] = variables[k].type; });
    (m.settings || []).forEach(function (s) { kinds[s.name] = s.type; });
    return (m.code || []).map(function (line) {
      return line.replace(/\{([^}]*)\}/g, function (all, key) {
        var raw = /\|raw$/.test(key), n = raw ? key.slice(0, -4) : key, v;
        if (n in r.used) { v = r.used[n]; }
        else if (n in r.env) { v = out(r.env[n], kinds[n]); }
        else { return all; }
        if (raw && typeof v === 'string') { return v; }
        return literal(v, kinds[n]);
      });
    }).join('\n');
  }

  var api = {
    load: load, loadUrl: loadUrl, models: models, estimate: estimate, plan: plan, code: code,
    candidates: candidates, evaluate: evaluate, version: 1
  };
  return api;
}));
