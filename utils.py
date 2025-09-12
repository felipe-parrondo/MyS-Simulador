import ast
import math
import re
from typing import Callable, List

# ==========================
# Utilidades seguras y ayudas
# ==========================

def replace_math_constants(expr: str) -> str:
    # Replace standalone 'e' with math.e, but not when it's part of a word
    expr = re.sub(r'\be\b', str(math.e), expr)
    # Replace standalone 'pi' with math.pi, but not when it's part of a word
    expr = re.sub(r'\bpi\b', str(math.pi), expr)
    return expr

def make_safe_func(expr: str) -> Callable[[float], float]:
    # Replace 'e' and 'pi' with their numerical values
    expr = replace_math_constants(expr)
    
    # Convert caret (^) to double asterisk (**) for Python syntax
    expr = re.sub(r'\^', '**', expr)
    
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    allowed_names.update({"abs": abs, "pow": pow})
    expr_ast = ast.parse(expr, mode='eval')
    for node in ast.walk(expr_ast):
        if isinstance(node, ast.Name):
            if node.id != 'x' and node.id not in allowed_names:
                raise ValueError(f"Nombre no permitido en expresión: {node.id}")
        elif isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Expression,
                               ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
                               ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Constant,
                               ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
                               ast.LtE, ast.GtE, ast.And, ast.Or, ast.BoolOp)):
            continue
        else:
            raise ValueError(f"Nodo AST no permitido: {type(node).__name__}")
    code = compile(expr_ast, '<string>', 'eval')
    def f(x: float) -> float:
        return eval(code, {'__builtins__': {}}, {**allowed_names, 'x': x})
    return f

def numerical_derivative(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)

# ==========================
# Heurística para sugerir g(x)
# ==========================

def sugerir_g_desde_f(expr_f: str, x0: float, make_func=make_safe_func) -> List[str]:
    """Genera candidatos simples para g(x) a partir de f(x)."""
    try:
        f = make_func(expr_f)
    except Exception:
        return []
    try:
        df0 = numerical_derivative(f, x0)
    except Exception:
        df0 = 1.0
    candidatos = []
    for lam in [1.0, 1.0/max(1e-6, abs(df0)), 0.5, 0.2, 0.1]:
        lam_txt = ("%g" % lam)
        candidatos.append(f"x - ({lam_txt})*({expr_f})")
    if expr_f.strip().startswith('x-') or expr_f.strip().startswith('x -'):
        try:
            candidatos.append(f"x - ({expr_f})")
        except Exception:
            pass
    vistos = set()
    unicos = []
    for c in candidatos:
        if c not in vistos:
            unicos.append(c)
            vistos.add(c)
    return unicos[:5]
