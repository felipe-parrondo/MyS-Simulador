from typing import Callable, Optional, List, Tuple, Literal
from utils import numerical_derivative
import math

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
        history.append((n, x, g(x), abs_err, rel_err))  # Ensure g(x) is included for graphing
        if abs_err < tol:
            history.append((n + 1, x_next, g(x_next), 0.0, 0.0))
            return x_next, history
        x = x_next
    return None, history

def punto_fijo_aitken(f: Callable[[float], float], g: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 50):
    history = []
    x = x0

    for n in range(max_iter):
        # Generate three consecutive terms for Aitken acceleration
        x1 = g(x)      # x_{n+1}
        x2 = g(x1)     # x_{n+2}

        # Apply Aitken's delta-squared acceleration
        denom = x2 - 2*x1 + x
        if abs(denom) < 1e-14:  # Numerical stability check
            x_acc = x2  # Fallback to regular iteration
        else:
            x_acc = x - (x1 - x)**2 / denom

        # Validate x_acc to ensure it's within a reasonable range
        if not (math.isfinite(x_acc) and abs(x_acc) < 1e10):
            raise ValueError(f"Numerical instability detected at iteration {n}: x_acc={x_acc}")

        # Calculate errors
        abs_err = abs(x_acc - x)
        rel_err = abs_err / abs(x_acc) if x_acc != 0 else float('inf')

        # Store iteration info
        history.append((n, x, x_acc, abs_err, rel_err))

        # Check convergence
        if abs_err < tol:
            # Verify that the result is actually a root of f(x) = 0
            f_val = f(x_acc)
            if abs(f_val) < tol * 10:  # Allow some tolerance for f(x)
                return x_acc, history

        x = x_acc  # Use accelerated value for next iteration

    return None, history

def simpson13_bloque(f,a,b):
    h=(b-a)/2; return (h/3)*(f(a)+4*f(a+h)+f(b))

def simpson38_bloque(f,a,b):
    h=(b-a)/3; return (3*h/8)*(f(a)+3*f(a+h)+3*f(a+2*h)+f(b))

def boole_bloque(f,a,b):
    h=(b-a)/4; return (2*h/45)*(7*f(a)+32*f(a+h)+12*f(a+2*h)+32*f(a+3*h)+7*f(b))

def integrar_nc_compuesto(f, a, b, n, metodo="Simpson 1/3"):
    if n <= 0 or b <= a:
        raise ValueError("Datos inválidos")
    h = (b - a) / n
    k = 0
    xk = a
    I = 0.0
    plan = []

    def safe_f(x):
        # Handle singularity at x = 0 for sin(x)/x
        if x == 0:
            return 1  # Limit of sin(x)/x as x approaches 0
        return f(x)

    def aplicar(m, tag):
        nonlocal I, k, xk, plan
        xi, xf = xk, xk + m * h
        if tag == "S13":
            Ik = simpson13_bloque(safe_f, xi, xf)
        elif tag == "S38":
            Ik = simpson38_bloque(safe_f, xi, xf)
        elif tag == "BOOLE":
            Ik = boole_bloque(safe_f, xi, xf)
        elif tag == "TRAP":
            Ik = (xf - xi) * (safe_f(xi) + safe_f(xf)) / 2  # bloque trapecio (1)
        elif tag == "MID":
            Ik = (xf - xi) * safe_f((xi + xf) / 2)  # bloque rect. medio (1)
        else:
            raise ValueError("Método desconocido")
        I += Ik
        plan.append((tag, xi, xf, m))
        xk = xf
        k += m

    if metodo == "Simpson 1/3":
        while k + 2 <= n:
            aplicar(2, "S13")
        if n - k == 1:  # reemplazo final 2 -> 3
            if plan and plan[-1][0] == "S13":
                # deshacer último bloque 1/3
                _, xi, xf, _ = plan.pop()
                k -= 2
                xk = xi
                I -= simpson13_bloque(safe_f, xi, xf)
            aplicar(3, "S38")
    elif metodo == "Simpson 3/8":
        while k + 3 <= n:
            aplicar(3, "S38")
        resto = n - k
        if resto == 2:
            aplicar(2, "S13")
        elif resto == 1:
            aplicar(2, "S13")
            aplicar(3, "S38")
    elif metodo == "Boole":
        while k + 4 <= n:
            aplicar(4, "BOOLE")
        resto = n - k
        if resto == 3:
            aplicar(3, "S38")
        elif resto == 2:
            aplicar(2, "S13")
        elif resto == 1:
            aplicar(3, "S38")
            aplicar(2, "S13")
    elif metodo == "Trapecio":
        for _ in range(n):
            aplicar(1, "TRAP")
    elif metodo == "Rectángulo Medio":
        for _ in range(n):
            aplicar(1, "MID")
    else:
        raise ValueError("Método no reconocido")

    return I, plan

Esquema = Literal["progresiva", "regresiva", "central"]

def derivada_diferencias_finitas(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    esquema: Esquema = "central",
    derivada: int = 1,
) -> Tuple[Optional[float], List[Tuple[int, float, str, int, float, float]]]:
    """
    Derivación numérica por diferencias finitas (discretas y paso uniforme h).

    Teoría usada (únicamente la del apunte):
    - 1ª derivada:
        * Progresiva:  f'(x_i) ≈ [f(x_{i+1}) - f(x_i)] / h
        * Regresiva:   f'(x_i) ≈ [f(x_i) - f(x_{i-1})] / h
        * Central:     f'(x_i) ≈ [f(x_{i+1}) - f(x_{i-1})] / (2h)
    - 2ª derivada:
        * Progresiva:  f''(x_i) ≈ [f(x_{i+2}) - 2 f(x_{i+1}) + f(x_i)] / h^2
        * Regresiva:   f''(x_i) ≈ [f(x_i) - 2 f(x_{i-1}) + f(x_{i-2})] / h^2
        * Central:     f''(x_i) ≈ [f(x_{i+1}) - 2 f(x_i) + f(x_{i-1})] / h^2

    Args:
        f: función escalar f(x).
        x: punto donde aproximar la derivada.
        h: paso (> 0).
        esquema: 'progresiva' | 'regresiva' | 'central'.
        derivada: 1 para f'(x), 2 para f''(x).

    Returns:
        (valor, history)
        - valor: aproximación numérica (o None si combinación inválida).
        - history: [(nivel, h, etiqueta, evals, aproximacion, err_est)], con un solo nivel.
    """
    if h <= 0:
        raise ValueError("h debe ser > 0")
    if esquema not in ("progresiva", "regresiva", "central"):
        raise ValueError("Esquema no soportado")
    if derivada not in (1, 2):
        raise ValueError("Solo se admite derivada 1 o 2")

    evals = 0
    approx = None

    if derivada == 1:
        if esquema == "progresiva":
            approx = (f(x + h) - f(x)) / h
            evals = 2
        elif esquema == "regresiva":
            approx = (f(x) - f(x - h)) / h
            evals = 2
        elif esquema == "central":
            approx = (f(x + h) - f(x - h)) / (2 * h)
            evals = 2

    elif derivada == 2:
        if esquema == "progresiva":
            approx = (f(x + 2*h) - 2.0 * f(x + h) + f(x)) / (h**2)
            evals = 3
        elif esquema == "regresiva":
            approx = (f(x) - 2.0 * f(x - h) + f(x - 2*h)) / (h**2)
            evals = 3
        elif esquema == "central":
            approx = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h**2)
            evals = 3

    etiqueta = f"{esquema}-d{derivada}"
    history = [(0, h, etiqueta, evals, float(approx), math.nan)]
    return approx, history
