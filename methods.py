from typing import Callable, Optional, List, Tuple
from utils import numerical_derivative

# ==========================
# Métodos numéricos
# ==========================

def metodo_biseccion(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = 50):
    if a >= b:
        raise ValueError("Se requiere a < b en bisección")
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, [(0, a, b, a, f(a), 0.0, 0.0)]
    if fb == 0:
        return b, [(0, a, b, b, f(b), 0.0, 0.0)]
    if fa * fb > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos (teorema de Bolzano)")
    history = []
    x_prev = None
    for n in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        if x_prev is None:
            abs_err = float('inf')
            rel_err = float('inf')
        else:
            abs_err = abs(m - x_prev)
            rel_err = abs_err / abs(m) if m != 0 else float('inf')
        history.append((n, a, b, m, fm, abs_err, rel_err))
        if abs(fm) < tol or abs_err < tol or (b - a) / 2 < tol:
            return m, history
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        x_prev = m
    return None, history

def newton_raphson(f: Callable[[float], float], x0: float, df: Optional[Callable[[float], float]] = None,
                   tol: float = 1e-8, max_iter: int = 50):
    history = []
    x = x0
    for n in range(max_iter):
        fx = f(x)
        dfx = df(x) if df is not None else numerical_derivative(f, x)
        if abs(dfx) < 1e-14:
            raise RuntimeError("Derivada cerca de cero; Newton puede fallar")
        x_next = x - fx / dfx
        abs_err = abs(x_next - x)
        rel_err = abs_err / abs(x_next) if x_next != 0 else float('inf')
        history.append((n, x, fx, dfx, abs_err, rel_err))
        if abs_err < tol:
            history.append((n + 1, x_next, f(x_next),
                            df(x_next) if df else numerical_derivative(f, x_next), 0.0, 0.0))
            return x_next, history
        x = x_next
    return None, history

def metodo_secante(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-8, max_iter: int = 50):
    history = []
    x_prev, x = x0, x1
    f_prev, f_x = f(x_prev), f(x)
    for n in range(max_iter):
        denom = (f_x - f_prev)
        if abs(denom) < 1e-14:
            raise RuntimeError("Denominador casi cero en Secante")
        x_next = x - f_x * (x - x_prev) / denom
        abs_err = abs(x_next - x)
        rel_err = abs_err / abs(x_next) if x_next != 0 else float('inf')
        history.append((n, x, x_next, abs_err, rel_err))
        if abs_err < tol:
            history.append((n + 1, x_next, x_next, 0.0, 0.0))
            return x_next, history
        x_prev, x = x, x_next
        f_prev, f_x = f_x, f(x)
    return None, history

def punto_fijo(g: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 50):
    history = []
    x = x0
    for n in range(max_iter):
        x_next = g(x)
        abs_err = abs(x_next - x)
        rel_err = abs_err / abs(x_next) if x_next != 0 else float('inf')
        history.append((n, x, x_next, abs_err, rel_err))
        if abs_err < tol:
            history.append((n + 1, x_next, g(x_next), 0.0, 0.0))
            return x_next, history
        x = x_next
    return None, history

def punto_fijo_aitken(g: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 50):
    history = []
    x = x0
    for n in range(max_iter):
        x1 = g(x)
        x2 = g(x1)
        denom = x2 - 2 * x1 + x
        x_acc = x2 - (x2 - x1) ** 2 / denom if denom != 0 else x2
        abs_err = abs(x_acc - x)
        rel_err = abs_err / abs(x_acc) if x_acc != 0 else float('inf')
        history.append((n, x, x_acc, abs_err, rel_err))
        if abs_err < tol:
            return x_acc, history
        x = x_acc
    return None, history
