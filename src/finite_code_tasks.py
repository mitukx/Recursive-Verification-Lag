"""Finite-domain expression tasks; generated text is interpreted, never exec/eval'd.

This restricted Python-expression track is a controlled code-LM pilot, not a
 general Python benchmark. Unsupported syntax is a scored failure, never dropped.
"""
import ast
import operator
import re
from dataclasses import dataclass

BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
       ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
CALL = {'abs': abs, 'min': min, 'max': max}


def extract_expression(text):
    fenced = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    text = fenced.group(1) if fenced else text
    if 'return ' in text:
        text = text.split('return ', 1)[1]
    return text.strip().splitlines()[0].strip().removeprefix('Expression:').strip().strip('`')


def parse_expression(text):
    if len(text) > 4096:
        raise ValueError('source too long')
    tree = ast.parse(extract_expression(text), mode='eval')
    nodes = list(ast.walk(tree))
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.BinOp,
               ast.UnaryOp, ast.USub, ast.UAdd, ast.Not, ast.IfExp,
               ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Call,
               *BIN, *CMP)
    if len(nodes) > 128 or any(not isinstance(n, allowed) for n in nodes):
        raise ValueError('unsupported expression')
    for n in nodes:
        if isinstance(n, ast.Name) and n.id not in {'x', *CALL}:
            raise ValueError('unknown name')
        if isinstance(n, ast.Constant) and (type(n.value) not in (int, bool) or abs(n.value) > 10000):
            raise ValueError('unsupported constant')
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in CALL or n.keywords or len(n.args) > 3):
            raise ValueError('unsupported call')
    return tree.body


def interpret(n, x, depth=0):
    if depth > 32:
        raise ValueError('expression too deep')
    ev = lambda a: interpret(a, x, depth + 1)
    if isinstance(n, ast.Constant): value = n.value
    elif isinstance(n, ast.Name) and n.id == 'x': value = x
    elif isinstance(n, ast.BinOp): value = BIN[type(n.op)](ev(n.left), ev(n.right))
    elif isinstance(n, ast.UnaryOp):
        v = ev(n.operand)
        value = -v if isinstance(n.op, ast.USub) else (+v if isinstance(n.op, ast.UAdd) else not v)
    elif isinstance(n, ast.IfExp): value = ev(n.body) if ev(n.test) else ev(n.orelse)
    elif isinstance(n, ast.Compare):
        left = ev(n.left); value = True
        for op, right in zip(n.ops, n.comparators):
            r = ev(right)
            if not CMP[type(op)](left, r): value = False; break
            left = r
    elif isinstance(n, ast.BoolOp):
        value = ev(n.values[0])
        for item in n.values[1:]:
            if isinstance(n.op, ast.And) and not value: break
            if isinstance(n.op, ast.Or) and value: break
            value = ev(item)
    elif isinstance(n, ast.Call): value = CALL[n.func.id](*[ev(a) for a in n.args])
    else: raise ValueError('unsupported node')
    if type(value) not in (int, bool) or abs(value) > 10**9:
        raise ValueError('out of bounds')
    return value


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    spec: str
    reference: str
    split: str = 'development'
    domain: tuple = tuple(range(-16, 17))
    public: tuple = (0, 1, 2, 3)

    def prompt(self):
        ref = parse_expression(self.reference)
        examples = ', '.join(f'{x} -> {interpret(ref, x)}' for x in self.public)
        return (f'Write a Python expression in the integer variable x. {self.spec} '
                f'The domain is all integers from -16 to 16. Examples: {examples}. '
                'Reply with ONLY one expression on one line, no explanation. '
                'Allowed: x, integer constants, + - * // %, comparisons, and/or/not, '
                'conditional expressions (a if condition else b), abs, min, max. '
                'No imports, loops, power, division /, or other function calls.')


def tasks():
    rows = [
        ('abs','absolute','Return the absolute value of x.', 'abs(x)'),
        ('clip','clipping','Clip x to the interval from -3 to 5.', 'min(5,max(-3,x))'),
        ('sign','sign','Return -1 for negative x, 0 for zero, and 1 for positive x.', '(1 if x>0 else -1) if x else 0'),
        ('parity','parity','Return 1 when x is odd, and 0 otherwise.', 'x%2'),
        ('ceil3','division','Return the ceiling of x divided by 3 as an integer.', '-((-x)//3)'),
        ('trunc3','division','Divide x by 3 and truncate toward zero.', 'x//3 if x>=0 else -((-x)//3)'),
        ('distance','distance','Return the distance of x from the closed interval [2,5].', 'max(2-x,0,x-5)'),
        ('piece','piecewise','Return x squared if x is negative; otherwise return x+1.', 'x*x if x<0 else x+1'),
        ('wrap','modular','Wrap x into [-2,2] modulo 5.', '(x+2)%5-2'),
        ('multiple','divisibility','Return 1 if x is divisible by 3 or 5, otherwise 0.', '1 if x%3==0 or x%5==0 else 0'),
        ('triangle','polynomial','Return x times (x+1) divided by 2.', 'x*(x+1)//2'),
        ('outside','interval','Return 1 if x is outside [-2,4], otherwise 0.', '1 if x<-2 or x>4 else 0'),
    ]
    return [Task(*r) for r in rows]


def score(text, task):
    ref = parse_expression(task.reference)
    try: tree = parse_expression(text)
    except (ValueError, SyntaxError, IndexError, RecursionError): tree = None
    passed = {}; executed = {}
    for x in task.domain:
        try:
            actual = interpret(tree, x)
            passed[x] = int(actual == interpret(ref, x)); executed[x] = 1
        except (ValueError, TypeError, ZeroDivisionError, KeyError, RecursionError):
            passed[x] = 0; executed[x] = 0
    public = sum(passed[x] for x in task.public)/len(task.public)
    hidden = [x for x in task.domain if x not in task.public]
    features = {'public_score': public, 'valid': int(tree is not None),
                'public_execution': sum(executed[x] for x in task.public)/len(task.public),
                'length_scaled': min(len(text),1024)/1024,
                'has_conditional': int(' if ' in text), 'has_mod': int('%' in text),
                'has_abs': int('abs' in text)}
    return {'public_score': public,
            'trusted_score': sum(passed[x] for x in hidden)/len(hidden),
            'exhaustive_score': sum(passed.values())/len(task.domain),
            'features': features}
