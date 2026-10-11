"""PyTomography's memory model, evaluated in pure Python (no torch): the same arithmetic as
``docs/source/_static/memory/pytomo-memory.js`` evaluates in the browser, from the same file,
``docs/source/_static/memory/model.json``.

The model re-expresses the ``estimate_memory()`` and ``plan_memory()`` methods of PyTomography's system matrices,
algorithms, filtered back projection and single scatter simulation as data: named parameters, derived quantities and
memory parts written in a small expression language, and the settings a plan searches. ``check_memory_model.py``
proves that this evaluator agrees with the library, and that the JavaScript evaluator agrees with this one.

Usage::

    from memory_model import MemoryModel
    model = MemoryModel.load()                       # docs/source/_static/memory/model.json
    model.estimate('pet_sinogram', {'tof': True}, {'n_subsets': 28})
    model.plan('pet_sinogram', {'tof': True}, ram_gb=25, gpu_gb=24)

Module-level ``estimate(model_id, params, settings)`` and ``plan(model_id, params, ram_gb, gpu_gb, fixed=None)`` use
the default model file.

The expression language (also documented at the top of model.json): numbers (``1.5e9``), strings in double or single
quotes, ``true``/``false`` (1 and 0), ``null``, ``pi``, names of parameters, derived quantities and settings; the
operators ``+ - * / % **`` (``+`` also joins strings; ``%`` takes the sign of the divisor, as in Python; ``**`` is right associative and binds
tighter than a unary minus on its left), comparisons ``< <= > >= == !=`` (giving 1 or 0), ``&& || !`` (giving 1 or 0),
``cond ? a : b`` and parentheses; the functions ``min max ceil floor round abs sqrt log2 log10 sin cos asin`` (``round`` to the nearest
integer, halves to even as Python's) and
``sig(x, n)`` (``x`` rounded to ``n`` significant digits, as Python's ``float(f'{x:.ng}')``). Nothing else is
evaluated: there is no access to Python objects.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from typing import Any, Mapping

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source', '_static', 'memory', 'model.json')


# ------------------------------------------------------------------------------------------------ expression language
class ExpressionError(ValueError):
    """An expression that does not parse or does not evaluate."""


_TWO_CHAR = ('**', '<=', '>=', '==', '!=', '&&', '||')
_ONE_CHAR = '+-*/%<>!?:(),'


def _tokens(text: str) -> list:
    """The tokens of an expression, as (kind, value): 'num', 'str', 'name' or 'op'."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or (c == '.' and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == '.'):
                j += 1
            if j < n and text[j] in 'eE':
                k = j + 1
                if k < n and text[k] in '+-':
                    k += 1
                if k < n and text[k].isdigit():
                    j = k
                    while j < n and text[j].isdigit():
                        j += 1
            out.append(('num', float(text[i:j])))
            i = j
        elif c.isalpha() or c == '_':
            j = i
            while j < n and (text[j].isalnum() or text[j] == '_'):
                j += 1
            out.append(('name', text[i:j]))
            i = j
        elif c in '"\'':
            j = text.find(c, i + 1)
            if j < 0:
                raise ExpressionError(f"unterminated string in {text!r}")
            out.append(('str', text[i + 1:j]))
            i = j + 1
        elif text[i:i + 2] in _TWO_CHAR:
            out.append(('op', text[i:i + 2]))
            i += 2
        elif c in _ONE_CHAR:
            out.append(('op', c))
            i += 1
        else:
            raise ExpressionError(f"unexpected character {c!r} in {text!r}")
    return out


class _Parser:
    """Recursive descent parser giving a tree of tuples: ('num', x), ('str', s), ('name', n), ('call', f, [args]),
    ('neg', a), ('not', a), ('bin', op, a, b), ('and', a, b), ('or', a, b), ('if', c, a, b)."""
    def __init__(self, text: str):
        self.text = text
        self.toks = _tokens(text)
        self.i = 0

    def peek(self, value=None):
        if self.i < len(self.toks):
            kind, v = self.toks[self.i]
            if value is None or (kind == 'op' and v == value):
                return self.toks[self.i]
        return None

    def take(self, value):
        if not self.peek(value):
            raise ExpressionError(f"expected {value!r} in {self.text!r}")
        self.i += 1

    def parse(self):
        tree = self.ternary()
        if self.i != len(self.toks):
            raise ExpressionError(f"unexpected {self.toks[self.i][1]!r} in {self.text!r}")
        return tree

    def ternary(self):
        cond = self.orr()
        if self.peek('?'):
            self.i += 1
            a = self.ternary()
            self.take(':')
            b = self.ternary()
            return ('if', cond, a, b)
        return cond

    def orr(self):
        a = self.andd()
        while self.peek('||'):
            self.i += 1
            a = ('or', a, self.andd())
        return a

    def andd(self):
        a = self.nott()
        while self.peek('&&'):
            self.i += 1
            a = ('and', a, self.nott())
        return a

    def nott(self):
        if self.peek('!'):
            self.i += 1
            return ('not', self.nott())
        return self.cmp()

    def cmp(self):
        a = self.add()
        for op in ('<=', '>=', '==', '!=', '<', '>'):
            if self.peek(op):
                self.i += 1
                return ('bin', op, a, self.add())
        return a

    def add(self):
        a = self.mul()
        while self.peek('+') or self.peek('-'):
            op = self.toks[self.i][1]
            self.i += 1
            a = ('bin', op, a, self.mul())
        return a

    def mul(self):
        a = self.unary()
        while self.peek('*') or self.peek('/') or self.peek('%'):
            op = self.toks[self.i][1]
            self.i += 1
            a = ('bin', op, a, self.unary())
        return a

    def unary(self):
        if self.peek('-'):
            self.i += 1
            return ('neg', self.unary())
        if self.peek('+'):
            self.i += 1
            return self.unary()
        return self.power()

    def power(self):
        a = self.atom()
        if self.peek('**'):
            self.i += 1
            return ('bin', '**', a, self.unary())
        return a

    def atom(self):
        tok = self.peek()
        if tok is None:
            raise ExpressionError(f"unexpected end of {self.text!r}")
        kind, v = tok
        self.i += 1
        if kind == 'num':
            return ('num', v)
        if kind == 'str':
            return ('str', v)
        if kind == 'name':
            if self.peek('('):
                self.i += 1
                args = []
                if not self.peek(')'):
                    args.append(self.ternary())
                    while self.peek(','):
                        self.i += 1
                        args.append(self.ternary())
                self.take(')')
                if v not in _FUNCTIONS:
                    raise ExpressionError(f"unknown function {v!r} in {self.text!r}")
                return ('call', v, args)
            return ('name', v)
        if v == '(':
            tree = self.ternary()
            self.take(')')
            return tree
        raise ExpressionError(f"unexpected {v!r} in {self.text!r}")


def _sig(x: float, n: float) -> float:
    """``x`` rounded to ``n`` significant digits, as Python's ``float(f'{x:.{n}g}')``."""
    return float(f"{x:.{int(n)}g}")


def _num(x):
    if x is None or isinstance(x, str):
        raise ExpressionError(f"a number was expected, got {x!r}")
    return float(x)


_FUNCTIONS = {
    'min': lambda *a: min(_num(x) for x in a),
    'max': lambda *a: max(_num(x) for x in a),
    'ceil': lambda x: float(math.ceil(_num(x))),
    'floor': lambda x: float(math.floor(_num(x))),
    'round': lambda x: float(round(_num(x))),
    'abs': lambda x: abs(_num(x)),
    'sqrt': lambda x: math.sqrt(_num(x)),
    'log2': lambda x: math.log2(_num(x)),
    'log10': lambda x: math.log10(_num(x)),
    'sin': lambda x: math.sin(_num(x)),
    'cos': lambda x: math.cos(_num(x)),
    'asin': lambda x: math.asin(_num(x)),
    'sig': lambda x, n: _sig(_num(x), _num(n)),
}

_CONSTANTS = {'true': 1.0, 'false': 0.0, 'null': None, 'pi': math.pi}


def _text(x) -> str:
    """A value joined to a string with ``+``: integers without a decimal point."""
    if x is None:
        return 'null'
    if isinstance(x, float) and x.is_integer() and abs(x) < 2 ** 53:
        return str(int(x))
    return str(x)


def _truthy(x) -> bool:
    return x is not None and x != 0 and x != ''


def _div(a: float, b: float) -> float:
    if b == 0:
        return math.nan if a == 0 else math.copysign(math.inf, a)
    return a / b


def _evaluate(tree, env: Mapping[str, Any]):
    kind = tree[0]
    if kind in ('num', 'str'):
        return tree[1]
    if kind == 'name':
        name = tree[1]
        if name in env:
            return env[name]
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        raise ExpressionError(f"unknown name {name!r}")
    if kind == 'neg':
        return -_num(_evaluate(tree[1], env))
    if kind == 'not':
        return 0.0 if _truthy(_evaluate(tree[1], env)) else 1.0
    if kind == 'and':
        return 1.0 if _truthy(_evaluate(tree[1], env)) and _truthy(_evaluate(tree[2], env)) else 0.0
    if kind == 'or':
        return 1.0 if _truthy(_evaluate(tree[1], env)) or _truthy(_evaluate(tree[2], env)) else 0.0
    if kind == 'if':
        return _evaluate(tree[2] if _truthy(_evaluate(tree[1], env)) else tree[3], env)
    if kind == 'call':
        return _FUNCTIONS[tree[1]](*(_evaluate(a, env) for a in tree[2]))
    op, a, b = tree[1], _evaluate(tree[2], env), _evaluate(tree[3], env)
    if op == '==':
        return 1.0 if a == b else 0.0
    if op == '!=':
        return 0.0 if a == b else 1.0
    if op == '+' and (isinstance(a, str) or isinstance(b, str)):
        return _text(a) + _text(b)
    a, b = _num(a), _num(b)
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op == '*':
        return a * b
    if op == '/':
        return _div(a, b)
    if op == '%':
        return a - b * math.floor(_div(a, b)) if b != 0 else math.nan
    if op == '**':
        return a ** b
    if op == '<':
        return 1.0 if a < b else 0.0
    if op == '<=':
        return 1.0 if a <= b else 0.0
    if op == '>':
        return 1.0 if a > b else 0.0
    if op == '>=':
        return 1.0 if a >= b else 0.0
    raise ExpressionError(f"unknown operator {op!r}")


_CACHE: dict = {}


def evaluate(expression, env: Mapping[str, Any]):
    """The value of ``expression`` (a string, or a number, boolean or None taken as it is) with the names in ``env``."""
    if expression is None or isinstance(expression, bool):
        return None if expression is None else float(expression)
    if isinstance(expression, (int, float)):
        return float(expression)
    tree = _CACHE.get(expression)
    if tree is None:
        tree = _CACHE[expression] = _Parser(expression).parse()
    return _evaluate(tree, env)


def _value(x):
    """A parameter or setting value as the expressions see it: booleans and integers as floats, strings and None kept."""
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float)):
        return float(x)
    return x


def _out(x, kind: str | None = None):
    """A value as estimates and plans report it: integers as int, booleans (settings of type 'bool') as bool."""
    if x is None or isinstance(x, str):
        return x
    if kind == 'bool':
        return bool(x)
    if isinstance(x, float) and x.is_integer() and abs(x) < 2 ** 53:
        return int(x)
    return x


# ---------------------------------------------------------------------------------------------------------- the model
class MemoryModel:
    """The model of ``model.json``: its parameters, derived quantities, models, overheads and plan searches."""

    def __init__(self, data: Mapping):
        self.data = data
        self.variables = {v['name']: v for v in data['variables']}
        self.derived_sets = data.get('derived_sets', {})
        self.overheads = data['overheads']
        self.by_id = {m['id']: m for m in data['models']}

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL) -> 'MemoryModel':
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------ description
    def _params_of(self, model: Mapping) -> list:
        defaults = model.get('defaults', {})
        params = []
        for name in model['params']:
            p = dict(self.variables[name])
            if name in defaults:
                p['default'] = defaults[name]
            params.append(p)
        return params

    def models(self) -> list:
        """Every model as ``{id, title, modality, params, settings}``, each parameter with its label, unit, default,
        range and options, as the page lists them."""
        out = []
        for m in self.data['models']:
            params = [{k: p[k] for k in ('name', 'label', 'unit', 'default', 'min', 'max', 'step', 'options', 'type', 'help') if k in p}
                      for p in self._params_of(m)]
            out.append({'id': m['id'], 'title': m['title'], 'modality': m['modality'], 'params': params,
                        'settings': [{'name': s['name'], 'label': s.get('label', s['name'])} for s in m.get('settings', [])]})
        return out

    # ------------------------------------------------------------------ evaluation
    def _env(self, model: Mapping, params: Mapping | None, settings: Mapping | None = None) -> dict:
        """Parameters (defaults where not given) with ``settings`` over them, and every derived quantity."""
        env = {}
        for p in self._params_of(model):
            env[p['name']] = _value(p.get('default'))
        for k, v in (params or {}).items():
            env[k] = _value(v)
        for k, v in (settings or {}).items():
            env[k] = _value(v)
        for group in self.data.get('derived', []), *(self.derived_sets[name] for name in model.get('use', [])), model.get('derived', []):
            for name, expression in (group.items() if isinstance(group, Mapping) else group):
                env[name] = evaluate(expression, env)
        return env

    def _active_settings(self, model: Mapping, env: Mapping) -> list:
        return [s for s in model.get('settings', []) if 'when' not in s or _truthy(evaluate(s['when'], env))]

    def _forced(self, setting: Mapping, env: Mapping):
        return evaluate(setting['force'], env) if 'force' in setting else None

    def _resolve(self, model: Mapping, params: Mapping | None, settings: Mapping | None) -> tuple:
        """The environment of an estimate, and the settings it is for (each as the model reports it)."""
        env0 = self._env(model, params)
        given = dict(settings or {})
        for s in self._active_settings(model, env0):
            forced = self._forced(s, env0)
            if forced is not None:
                given[s['name']] = forced
        env = self._env(model, params, given)
        used = {}
        for s in self._active_settings(model, env):
            used[s['name']] = _out(evaluate(s.get('report', s['name']), env), s.get('type'))
        return env, used

    def _parts(self, model: Mapping, env: Mapping) -> list:
        """The model's own parts, as (name, scope, ram bytes, gpu bytes)."""
        parts = []
        for p in self._expand(model['parts']):
            if 'repeat' in p:
                r = p['repeat']
                lo, hi = int(_num(evaluate(r['from'], env))), int(_num(evaluate(r['to'], env)))
                local = dict(env)
                for k in range(lo, hi + 1):
                    local[r['var']] = float(k)
                    self._part(p, local, parts)
            else:
                self._part(p, env, parts)
        return parts

    def _part(self, p: Mapping, env: Mapping, parts: list) -> None:
        """Adds part ``p`` as (name, scope, ram bytes, gpu bytes) to ``parts``, unless its ``when`` is false."""
        if 'when' in p and not _truthy(evaluate(p['when'], env)):
            return
        if 'bytes' in p:
            nbytes = _num(evaluate(p['bytes'], env))
            on_gpu = _truthy(evaluate(p.get('on', 0), env))
            ram, gpu = (0.0, nbytes) if on_gpu else (nbytes, 0.0)
        else:
            ram = _num(evaluate(p.get('ram', 0), env))
            gpu = _num(evaluate(p.get('gpu', 0), env))
        parts.append((self._name(p['name'], env), p.get('scope', 'held'), ram, gpu))

    def _expand(self, parts: list) -> list:
        """The parts with every ``{'include': name}`` replaced by the parts of ``part_sets[name]``."""
        out = []
        for p in parts:
            if 'include' in p:
                out.extend(self._expand(self.data['part_sets'][p['include']]))
            else:
                out.append(p)
        return out

    @staticmethod
    def _name(template: str, env: Mapping) -> str:
        """A part's name with ``{expression}`` replaced by its value."""
        out, i = [], 0
        while True:
            j = template.find('{', i)
            if j < 0:
                out.append(template[i:])
                return ''.join(out)
            k = template.find('}', j)
            out.append(template[i:j])
            value = _out(evaluate(template[j + 1:k], env))
            out.append(str(value))
            i = k + 1

    def _with_overheads(self, model: Mapping, env: Mapping, parts: list) -> list:
        """The parts with the overheads every estimate adds (``overheads`` of model.json)."""
        o = self.overheads
        out = list(parts)
        out.append((o['fixed']['name'], 'overhead', _num(evaluate(o['fixed']['bytes'], env)), 0.0))
        gpu_arrays = sum(p[3] for p in parts)
        if gpu_arrays > 0:
            out.append((o['cuda_cache']['name'], 'overhead', 0.0, o['cuda_cache']['bytes'] + o['cuda_cache']['fraction'] * gpu_arrays))
        if _truthy(evaluate(o['gpu_counts_as_ram'], env)):
            arrays = sum(p[2] + p[3] for p in parts)
            fraction = model.get('allocator_fraction', o['allocator']['fraction'])
            if arrays > 0:
                out.append((o['allocator']['name'], 'overhead', fraction * arrays, 0.0))
        return out

    def _totals(self, env: Mapping, parts: list) -> tuple:
        gpu = sum(p[3] for p in parts)
        ram = sum(p[2] for p in parts)
        if _truthy(evaluate(self.overheads['gpu_counts_as_ram'], env)):
            ram += gpu
        return ram / 1e9, gpu / 1e9

    def estimate(self, model_id: str, params: Mapping | None = None, settings: Mapping | None = None) -> dict:
        """The predicted peak of model ``model_id`` with ``params`` and ``settings`` (defaults for those not given), as
        ``{ram_gb, gpu_gb, parts: [{name, scope, ram_gb, gpu_gb}], settings}``. On Windows ``ram_gb`` includes the GPU
        memory."""
        model = self.by_id[model_id]
        env, used = self._resolve(model, params, settings)
        parts = self._with_overheads(model, env, self._parts(model, env))
        ram_gb, gpu_gb = self._totals(env, parts)
        return {'model': model_id, 'ram_gb': ram_gb, 'gpu_gb': gpu_gb, 'settings': used,
                'parts': [{'name': n, 'scope': s, 'ram_gb': r / 1e9, 'gpu_gb': g / 1e9} for n, s, r, g in parts]}

    def candidates(self, model_id: str, params: Mapping | None = None) -> dict:
        """The values a plan tries for each setting, fastest first, in the order they are searched."""
        model = self.by_id[model_id]
        env = self._env(model, params)
        out = {}
        for s in self._active_settings(model, env):
            out[s['name']] = [_out(v, s.get('type')) for v in self._candidates(s, env)]
        return out

    def _candidates(self, setting: Mapping, env: Mapping) -> list:
        forced = self._forced(setting, env)
        if forced is not None:
            return [forced]
        spec = setting['candidates']
        values = []
        for item in spec['items']:
            if 'value' in item:
                local = env
                if 'if' not in item or _truthy(evaluate(item['if'], local)):
                    values.append(evaluate(item['value'], local))
            else:
                seq = item['seq']
                lo, hi = int(_num(evaluate(seq['from'], env))), int(_num(evaluate(seq['to'], env)))
                local = dict(env)
                for k in range(lo, hi + 1):
                    local[seq['var']] = float(k)
                    if 'if' in seq:
                        local['value'] = evaluate(seq['value'], local)
                        if not _truthy(evaluate(seq['if'], local)):
                            continue
                    values.append(evaluate(seq['value'], local))
        if spec.get('unique'):
            unique = []
            for v in values:
                if v not in unique:
                    unique.append(v)
            values = unique
        if spec.get('sort') == 'asc':
            values = sorted(values)
        return values

    def plan(self, model_id: str, params: Mapping | None = None, ram_gb: float = math.inf, gpu_gb: float = math.inf,
             fixed: Mapping | None = None) -> dict:
        """The fastest settings of model ``model_id`` whose estimate fits ``ram_gb`` of RAM and ``gpu_gb`` of GPU memory,
        searched as the library's ``plan_settings``: every combination of the candidates, the first setting the outer
        loop, each fastest first; the first that fits, or the one that exceeds the budgets least. Settings in ``fixed``
        (not None) keep their value. Returns ``{settings, estimate, fits, options: [{settings, ram_gb, gpu_gb, fits}]}``."""
        model = self.by_id[model_id]
        env = self._env(model, params)
        fixed = {k: v for k, v in (fixed or {}).items() if v is not None}
        active = self._active_settings(model, env)
        names = [s['name'] for s in active]
        unknown = sorted(set(fixed) - set(names))
        if unknown:
            raise ValueError(f"{model_id} has no setting {', '.join(unknown)}: its settings are {', '.join(names) or 'none'}")
        lists, kept = [], []
        for s in active:
            forced = self._forced(s, env)
            if forced is not None:
                values = [forced]
                kept.append(s['name'])
            elif s['name'] in fixed:
                values = [_value(fixed[s['name']])]
                kept.append(s['name'])
            else:
                values = self._candidates(s, env) or [env.get(s['name'])]
            lists.append(values)
        types = {s['name']: s.get('type') for s in active}
        options, chosen, closest, least = [], None, None, math.inf
        for values in itertools.product(*lists):
            trial = dict(zip(names, values))
            e = self.estimate(model_id, params, trial)
            excess = max(0.0, e['ram_gb'] - ram_gb) + max(0.0, e['gpu_gb'] - gpu_gb)
            fits = excess <= 1e-9
            options.append({'settings': {k: _out(v, types[k]) for k, v in trial.items()}, 'ram_gb': e['ram_gb'], 'gpu_gb': e['gpu_gb'], 'fits': fits})
            if fits:
                chosen = trial
                break
            if excess < least:
                closest, least = trial, excess
        settings = chosen if chosen is not None else closest
        estimate = self.estimate(model_id, params, settings)
        return {'model': model_id, 'settings': {k: _out(v, types[k]) for k, v in settings.items()}, 'estimate': estimate,
                'fits': chosen is not None, 'options': options, 'ram_gb': ram_gb, 'gpu_gb': gpu_gb,
                'fixed': [n for n in names if n in kept]}

    # ------------------------------------------------------------------ code
    def code(self, model_id: str, params: Mapping | None = None, settings: Mapping | None = None) -> str:
        """A short Python snippet that sets these settings in PyTomography and prints the library's own estimate."""
        model = self.by_id[model_id]
        env, used = self._resolve(model, params, settings)
        kinds = {s['name']: s.get('type') for s in model.get('settings', [])}
        kinds.update({name: v.get('type') for name, v in self.variables.items() if name not in kinds})

        def literal(key: str) -> str:
            raw = key.endswith('|raw')
            name = key[:-4] if raw else key
            if name in used:
                value = used[name]
            elif name in env:
                value = _out(env[name], kinds.get(name))
            else:
                return '{' + key + '}'
            if raw and isinstance(value, str):
                return value
            if value is None:
                return 'None'
            if isinstance(value, bool) or kinds.get(name) == 'bool':
                return 'True' if value else 'False'
            if isinstance(value, str):
                return "'" + value + "'"
            return str(value) if isinstance(value, int) else repr(value)

        lines = []
        for line in model.get('code', []):
            out, i = [], 0
            while True:
                j = line.find('{', i)
                if j < 0:
                    out.append(line[i:])
                    break
                k = line.find('}', j)
                out.append(line[i:j])
                out.append(literal(line[j + 1:k]))
                i = k + 1
            lines.append(''.join(out))
        return '\n'.join(lines)


_DEFAULT: MemoryModel | None = None


def _default() -> MemoryModel:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MemoryModel.load()
    return _DEFAULT


def estimate(model_id: str, params: Mapping | None = None, settings: Mapping | None = None) -> dict:
    """:meth:`MemoryModel.estimate` of the default model file."""
    return _default().estimate(model_id, params, settings)


def plan(model_id: str, params: Mapping | None = None, ram_gb: float = math.inf, gpu_gb: float = math.inf,
         fixed: Mapping | None = None) -> dict:
    """:meth:`MemoryModel.plan` of the default model file."""
    return _default().plan(model_id, params, ram_gb, gpu_gb, fixed)


if __name__ == '__main__':
    import sys
    model = MemoryModel.load(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    for m in model.models():
        e = model.estimate(m['id'])
        print(f"{m['id']:24s} {e['ram_gb']:8.2f} GB RAM {e['gpu_gb']:8.2f} GB GPU  {e['settings']}")
