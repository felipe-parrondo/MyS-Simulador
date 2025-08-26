import ast
import math
import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional, List, Tuple, Dict

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:
    FigureCanvasTkAgg = None
    plt = None
    FuncAnimation = None

# ==========================
# Utilidades seguras
# ==========================

def _make_safe_func(expr: str) -> Callable[[float], float]:
    """Compila una expresión en una función segura f(x)."""
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

# ==========================
# Heurística para sugerir g(x)
# ==========================

def sugerir_g_desde_f(expr_f: str, x0: float) -> List[str]:
    """Genera candidatos simples para g(x) a partir de f(x)."""
    try:
        f = _make_safe_func(expr_f)
    except Exception:
        return []
    # Derivada estimada en x0
    try:
        df0 = numerical_derivative(f, x0)
    except Exception:
        df0 = 1.0
    candidatos = []
    # 1) g(x) = x - lambda f(x)
    for lam in [1.0, 1.0/max(1e-6, abs(df0)), 0.5, 0.2, 0.1]:
        # Redondeo amigable del lambda
        lam_txt = ("%g" % lam)
        candidatos.append(f"x - ({lam_txt})*({expr_f})")
    # 2) Si f(x) ~ x - h(x) => g(x)=h(x); intentamos aislar x si forma 'x - (...)'
    if expr_f.strip().startswith('x-') or expr_f.strip().startswith('x -'):
        try:
            rhs = expr_f.strip()[1:]  # quita el primer 'x'
            # g(x) = x - f(x)  => si f(x)=x-h(x), g(x)=h(x)
            candidatos.append(f"x - ({expr_f})")
        except Exception:
            pass
    # Filtrar duplicados simples
    vistos = set()
    unicos = []
    for c in candidatos:
        if c not in vistos:
            unicos.append(c)
            vistos.add(c)
    return unicos[:5]

# ==========================
# GUI
# ==========================

class SimuladorRaices:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Laboratorio de Métodos Numéricos – Raíces")
        master.geometry("1050x700")

        self.current_method = tk.StringVar(value="Newton-Raphson")
        self.decimals = tk.IntVar(value=6)
        self.modo_estudiante = tk.BooleanVar(value=False)

        self.historia_actual: List[Tuple] = []
        self.historia_comparacion: Dict[str, List[Tuple]] = {}
        self.ultimo_resultado: Optional[float] = None

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        cont = ttk.Frame(self.master, padding=8)
        cont.pack(fill=tk.BOTH, expand=True)

        # Barra superior
        top = ttk.Frame(cont)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Método:").pack(side=tk.LEFT)
        self.metodo_cb = ttk.Combobox(top, textvariable=self.current_method, state="readonly",
                                      values=["Newton-Raphson", "Secante", "Bisección", "Punto Fijo", "Punto Fijo + Aitken"]) 
        self.metodo_cb.pack(side=tk.LEFT, padx=6)
        self.metodo_cb.bind("<<ComboboxSelected>>", lambda e: self._update_table_headers())

        ttk.Label(top, text="Decimales:").pack(side=tk.LEFT, padx=(12, 0))
        tk.Spinbox(top, from_=2, to=15, textvariable=self.decimals, width=5).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(top, text="Modo estudiante (explicaciones)", variable=self.modo_estudiante).pack(side=tk.LEFT, padx=12)

        # Notebook
        nb = ttk.Notebook(cont)
        nb.pack(fill=tk.BOTH, expand=True, pady=6)
        self.nb = nb

        # ---- Tab Funciones ----
        tab_fun = ttk.Frame(nb)
        nb.add(tab_fun, text="Funciones y Parámetros")

        grid = ttk.Frame(tab_fun)
        grid.pack(fill=tk.X, pady=6)

        # f(x), f'(x), g(x)
        ttk.Label(grid, text="f(x):").grid(row=0, column=0, sticky='w')
        self.expr_f = tk.StringVar(value="x**2 - 2")
        ttk.Entry(grid, textvariable=self.expr_f, width=50).grid(row=0, column=1, columnspan=4, sticky='we')

        ttk.Label(grid, text="f'(x) (opcional):").grid(row=1, column=0, sticky='w')
        self.expr_df = tk.StringVar(value="")
        ttk.Entry(grid, textvariable=self.expr_df, width=50).grid(row=1, column=1, columnspan=4, sticky='we')

        ttk.Label(grid, text="g(x) (punto fijo):").grid(row=2, column=0, sticky='w')
        self.expr_g = tk.StringVar(value="(x + 2/x)/2")
        ttk.Entry(grid, textvariable=self.expr_g, width=50).grid(row=2, column=1, columnspan=4, sticky='we')

        # Parámetros numéricos
        ttk.Label(grid, text="x0:").grid(row=3, column=0, sticky='w')
        self.var_x0 = tk.StringVar(value="1.5")
        ttk.Entry(grid, textvariable=self.var_x0, width=10).grid(row=3, column=1, sticky='w')

        ttk.Label(grid, text="x1 (Secante):").grid(row=3, column=2, sticky='w')
        self.var_x1 = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.var_x1, width=10).grid(row=3, column=3, sticky='w')

        ttk.Label(grid, text="Intervalo [a,b] (Bisección):").grid(row=4, column=0, sticky='w')
        self.var_a = tk.StringVar(value="0.0")
        self.var_b = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.var_a, width=10).grid(row=4, column=1, sticky='w')
        ttk.Entry(grid, textvariable=self.var_b, width=10).grid(row=4, column=2, sticky='w')

        ttk.Label(grid, text="tol:").grid(row=5, column=0, sticky='w')
        self.var_tol = tk.StringVar(value="1e-8")
        ttk.Entry(grid, textvariable=self.var_tol, width=10).grid(row=5, column=1, sticky='w')

        ttk.Label(grid, text="max iter:").grid(row=5, column=2, sticky='w')
        self.var_max = tk.StringVar(value="50")
        ttk.Entry(grid, textvariable=self.var_max, width=10).grid(row=5, column=3, sticky='w')

        # Botones de acción
        btns = ttk.Frame(tab_fun)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="Ejecutar", command=self.ejecutar).pack(side=tk.LEFT)
        ttk.Button(btns, text="Comparar métodos", command=self.comparar_metodos).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Graficar", command=self.graficar).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Animar", command=self.animar).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Guardar CSV", command=self.guardar_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Guardar gráfica", command=self.guardar_png).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Guardar animación (.mp4)", command=self.guardar_animacion).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Sugerir g(x)", command=self.sugerir_g).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Limpiar", command=self.limpiar).pack(side=tk.LEFT, padx=6)

        # ---- Tab Tabla ----
        tab_tabla = ttk.Frame(nb)
        nb.add(tab_tabla, text="Tabla de iteraciones")

        self.tree = ttk.Treeview(tab_tabla, show='headings', height=18)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self._update_table_headers()

        # ---- Tab Gráficas ----
        tab_plot = ttk.Frame(nb)
        nb.add(tab_plot, text="Gráficas")

        if FigureCanvasTkAgg and plt:
            self.fig, self.ax = plt.subplots(figsize=(7.5, 4.5))
            self.canvas = FigureCanvasTkAgg(self.fig, master=tab_plot)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.ax = None
            self.canvas = None
            ttk.Label(tab_plot, text="matplotlib no disponible").pack(pady=12)

        # ---- Tab Ayuda ----
        tab_help = ttk.Frame(nb)
        nb.add(tab_help, text="Ayuda / Teoría")
        help_txt = tk.Text(tab_help, wrap='word', height=20)
        help_txt.pack(fill=tk.BOTH, expand=True)
        help_txt.insert(tk.END, self._texto_ayuda())
        help_txt.configure(state=tk.DISABLED)

        # Estado inferior
        self.lbl_estado = ttk.Label(cont, text="Listo")
        self.lbl_estado.pack(fill=tk.X, pady=(4, 0))

    def _texto_ayuda(self) -> str:
        return (
            "Métodos disponibles:\n"
            "• Bisección: requiere [a,b] con cambio de signo. Convergencia lineal.\n"
            "• Newton-Raphson: requiere f(x) y opcional f'(x). Convergencia cuadrática cerca de la raíz.\n"
            "• Secante: no requiere derivada. Convergencia superlineal.\n"
            "• Punto Fijo: requiere g(x). Converge si |g'(x*)|<1 (contracción).\n"
            "• Aitken: acelera la convergencia del punto fijo.\n\n"
            "Sugerir g(x): propone formas g(x)=x-λ f(x) con λ heurístico.\n"
            "Exportación: guarda tabla (CSV), gráfica (PNG) y animación (MP4).\n"
        )

    def _update_table_headers(self):
        method = self.current_method.get()
        if method == "Newton-Raphson":
            cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "err_abs", "err_rel")
        elif method == "Secante":
            cols = ("n", "x_n", "x_{n+1}", "err_abs", "err_rel")
        elif method == "Bisección":
            cols = ("n", "a", "b", "m", "f(m)", "err_abs", "err_rel")
        else:  # Punto Fijo (+ Aitken)
            cols = ("n", "x_n", "g(x_n)", "err_abs", "err_rel")
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor='center')

    def _llenar_tabla(self, history: List[Tuple]):
        for row in self.tree.get_children():
            self.tree.delete(row)
        d = self.decimals.get()
        for rec in history:
            vals = []
            for v in rec:
                if isinstance(v, float):
                    vals.append(f"{v:.{d}g}")
                else:
                    vals.append(str(v))
            self.tree.insert('', 'end', values=tuple(vals))

    # ---------- Lógica ----------
    def ejecutar(self):
        try:
            metodo = self.current_method.get()
            f = _make_safe_func(self.expr_f.get())
            df = _make_safe_func(self.expr_df.get()) if self.expr_df.get().strip() else None
            g = _make_safe_func(self.expr_g.get())
            x0 = float(self.var_x0.get())
            x1 = float(self.var_x1.get())
            a = float(self.var_a.get())
            b = float(self.var_b.get())
            tol = float(self.var_tol.get())
            max_iter = int(self.var_max.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        try:
            if metodo == "Bisección":
                root, hist = metodo_biseccion(f, a, b, tol, max_iter)
            elif metodo == "Newton-Raphson":
                root, hist = newton_raphson(f, x0, df, tol, max_iter)
            elif metodo == "Secante":
                root, hist = metodo_secante(f, x0, x1, tol, max_iter)
            elif metodo == "Punto Fijo":
                self._verificar_contraccion(g, x0)
                root, hist = punto_fijo(g, x0, tol, max_iter)
            else:  # Aitken
                self._verificar_contraccion(g, x0)
                root, hist = punto_fijo_aitken(g, x0, tol, max_iter)
        except Exception as e:
            messagebox.showerror("Error en ejecución", str(e))
            return

        self.historia_actual = hist
        self.ultimo_resultado = root
        self._llenar_tabla(hist)

        if root is not None:
            self._status_ok(f"Convergió a {root:.6g}")
        else:
            self._status_err("No convergió con los parámetros dados")
        self.graficar()

    def comparar_metodos(self):
        """Corre Newton, Secante, Punto Fijo y Aitken (si aplicables) y grafica en conjunto."""
        try:
            f = _make_safe_func(self.expr_f.get())
            df = _make_safe_func(self.expr_df.get()) if self.expr_df.get().strip() else None
            g = _make_safe_func(self.expr_g.get())
            x0 = float(self.var_x0.get())
            x1 = float(self.var_x1.get())
            tol = float(self.var_tol.get())
            max_iter = int(self.var_max.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        resultados = {}
        historias = {}

        try:
            rn, hn = newton_raphson(f, x0, df, tol, max_iter)
            resultados["Newton"] = rn
            historias["Newton"] = hn
        except Exception as e:
            self._status_err(f"Newton falló: {e}")

        try:
            rs, hs = metodo_secante(f, x0, x1, tol, max_iter)
            resultados["Secante"] = rs
            historias["Secante"] = hs
        except Exception as e:
            self._status_err(f"Secante falló: {e}")

        try:
            self._verificar_contraccion(g, x0, solo_warn=True)
            rpf, hpf = punto_fijo(g, x0, tol, max_iter)
            resultados["Punto Fijo"] = rpf
            historias["Punto Fijo"] = hpf
        except Exception as e:
            self._status_err(f"Punto fijo falló: {e}")

        try:
            self._verificar_contraccion(g, x0, solo_warn=True)
            ra, ha = punto_fijo_aitken(g, x0, tol, max_iter)
            resultados["Aitken"] = ra
            historias["Aitken"] = ha
        except Exception as e:
            self._status_err(f"Aitken falló: {e}")

        self.historia_comparacion = historias

        # Construir tabla resumen en la pestaña de tabla (encima de la actual)
        self._mostrar_resumen_comparacion(resultados, historias)
        self._grafica_comparacion(f, g, historias)
        self.nb.select(2)  # ir a Gráficas

    def _mostrar_resumen_comparacion(self, resultados, historias):
        # Limpiar tabla y poner columnas de resumen
        for row in self.tree.get_children():
            self.tree.delete(row)
        cols = ("Método", "Raíz", "Iteraciones")
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160, anchor='center')
        for m, r in resultados.items():
            iters = len(historias.get(m, []))
            r_txt = "-" if r is None else f"{r:.6g}"
            self.tree.insert('', 'end', values=(m, r_txt, iters))

    def _grafica_comparacion(self, f, g, historias):
        if not (self.ax and self.canvas):
            return
        # Determinar rango x a partir de todas las historias
        xs = []
        for h in historias.values():
            for rec in h:
                if len(rec) > 1 and isinstance(rec[1], (int, float)):
                    xs.append(float(rec[1]))
        if not xs:
            xs = [float(self.var_x0.get())]
        xmin, xmax = min(xs) - 1, max(xs) + 1
        X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]

        self.ax.clear()
        # f(x)
        try:
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, label='f(x)')
            self.ax.axhline(0, color='k', ls='--')
        except Exception:
            pass

        # Iteraciones por método
        for metodo, hist in historias.items():
            xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
            if not xs_plot:
                continue
            try:
                self.ax.plot(xs_plot, [f(x) for x in xs_plot], 'o-', label=f"Iteraciones {metodo}")
            except Exception:
                pass
        self.ax.legend()
        self.canvas.draw()

    def graficar(self):
        if not (self.ax and self.canvas):
            return
        metodo = self.current_method.get()
        try:
            f = _make_safe_func(self.expr_f.get())
            g = _make_safe_func(self.expr_g.get())
            x0 = float(self.var_x0.get())
        except Exception:
            return

        hist = self.historia_actual
        self.ax.clear()
        if not hist:
            # graficar sólo f/g alrededor de x0
            xmin, xmax = x0 - 5, x0 + 5
        else:
            xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
            xmin, xmax = (min(xs_plot) - 1, max(xs_plot) + 1) if xs_plot else (x0 - 5, x0 + 5)
        X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]

        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            try:
                Yg = [g(x) for x in X]
                self.ax.plot(X, Yg, label='g(x)')
                self.ax.plot(X, X, '--', label='y=x')
                if hist:
                    xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
                    self.ax.plot(xs_plot, [g(x) for x in xs_plot], 'o-', label='Iteraciones')
                if self.ultimo_resultado is not None:
                    self.ax.plot(self.ultimo_resultado, self.ultimo_resultado, 'ro', markersize=8, label='Punto fijo estimado')
            except Exception:
                pass
        else:
            try:
                Y = [f(x) for x in X]
                self.ax.plot(X, Y, label='f(x)')
                self.ax.axhline(0, color='k', ls='--')
                if hist:
                    xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
                    self.ax.plot(xs_plot, [f(x) for x in xs_plot], 'o-', label='Iteraciones')
                if self.ultimo_resultado is not None:
                    self.ax.plot(self.ultimo_resultado, 0, 'ro', markersize=8, label='Raíz estimada')
            except Exception:
                pass
        self.ax.legend()
        self.canvas.draw()

    def animar(self):
        if not (self.ax and self.canvas and FuncAnimation):
            messagebox.showerror("Error", "Animación no disponible (matplotlib.animation)")
            return
        hist = self.historia_actual
        if not hist:
            messagebox.showwarning("Atención", "Ejecuta primero un método para generar iteraciones")
            return
        metodo = self.current_method.get()
        try:
            f = _make_safe_func(self.expr_f.get())
            g = _make_safe_func(self.expr_g.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Preparar figura
        self.ax.clear()
        xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
        xmin, xmax = (min(xs_plot) - 1, max(xs_plot) + 1) if xs_plot else (-5, 5)
        X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]
        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            Yg = [g(x) for x in X]
            self.ax.plot(X, Yg, label='g(x)')
            self.ax.plot(X, X, '--', label='y=x')
        else:
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, label='f(x)')
            self.ax.axhline(0, color='k', ls='--')
        linea_iter, = self.ax.plot([], [], 'o-', label='Iteraciones')
        self.ax.legend()

        # Datos para animación
        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            ys = [g(x) for x in xs_plot]
        else:
            ys = [f(x) for x in xs_plot]

        def init():
            linea_iter.set_data([], [])
            return (linea_iter,)

        def update(i):
            linea_iter.set_data(xs_plot[:i+1], ys[:i+1])
            return (linea_iter,)

        self.anim = FuncAnimation(self.fig, update, init_func=init, frames=len(xs_plot), interval=600, blit=True)
        self.canvas.draw()
        self.nb.select(2)

    # ---------- Utilidades ----------
    def guardar_csv(self):
        if not self.historia_actual and not self.historia_comparacion:
            messagebox.showerror("Error", "No hay datos para guardar")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV","*.csv")])
        if not file_path:
            return
        try:
            with open(file_path, 'w', newline='') as f:
                w = csv.writer(f)
                if self.historia_comparacion:
                    w.writerow(["Resumen de comparación"])
                    w.writerow(["Método", "x", "Iteraciones"])
                    for m, hist in self.historia_comparacion.items():
                        x = None
                        if hist:
                            # último x conocido en historia
                            for rec in reversed(hist):
                                if isinstance(rec[1], (int, float)):
                                    x = rec[1]
                                    break
                        w.writerow([m, x, len(hist)])
                    w.writerow([])
                if self.historia_actual:
                    w.writerow(["Historia método actual"])
                    for rec in self.historia_actual:
                        w.writerow(rec)
            messagebox.showinfo("Éxito", f"CSV guardado en {file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def guardar_png(self):
        if not (self.ax and self.canvas):
            messagebox.showerror("Error", "matplotlib no disponible")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png"), ("PDF","*.pdf")])
        if not file_path:
            return
        try:
            self.fig.savefig(file_path, bbox_inches='tight', dpi=150)
            messagebox.showinfo("Éxito", f"Gráfica guardada en {file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def guardar_animacion(self):
        if not hasattr(self, 'anim') or self.anim is None:
            messagebox.showerror("Error", "Primero crea la animación con el botón 'Animar'")
            return
        if not FuncAnimation:
            messagebox.showerror("Error", "Animación no disponible")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4","*.mp4")])
        if not file_path:
            return
        try:
            # Requiere ffmpeg instalado en el sistema
            self.anim.save(file_path)
            messagebox.showinfo("Éxito", f"Animación guardada en {file_path}")
        except Exception as e:
            messagebox.showerror("Error al exportar MP4", str(e))

    def limpiar(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.historia_actual = []
        self.historia_comparacion = {}
        self.ultimo_resultado = None
        if self.ax and self.canvas:
            self.ax.clear()
            self.canvas.draw()
        self._status_ok("Limpio")

    def sugerir_g(self):
        try:
            x0 = float(self.var_x0.get())
            cands = sugerir_g_desde_f(self.expr_f.get(), x0)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if not cands:
            messagebox.showwarning("Sugerencias", "No se pudieron generar sugerencias para g(x)")
            return
        # Mostrar un diálogo simple para elegir
        dlg = tk.Toplevel(self.master)
        dlg.title("Sugerencias para g(x)")
        ttk.Label(dlg, text="Elige una forma sugerida para g(x):").pack(padx=10, pady=8)
        lb = tk.Listbox(dlg, height=min(8, len(cands)), width=60)
        for c in cands:
            lb.insert(tk.END, c)
        lb.pack(padx=10, pady=6)
        def aplicar():
            sel = lb.curselection()
            if sel:
                self.expr_g.set(lb.get(sel[0]))
            dlg.destroy()
        ttk.Button(dlg, text="Usar selección", command=aplicar).pack(pady=8)

    def _verificar_contraccion(self, g: Callable[[float], float], x0: float, solo_warn: bool=False):
        # Evalúa |g'(x)| en un pequeño vecindario de x0
        xs = [x0 + dx for dx in (-0.5, -0.25, 0, 0.25, 0.5)]
        vals = []
        for x in xs:
            try:
                vals.append(abs(numerical_derivative(g, x)))
            except Exception:
                pass
        if not vals:
            return
        max_mod = max(vals)
        msg = f"Máx |g'(x)| cerca de x0 ≈ {max_mod:.3g}. "
        if max_mod < 1:
            self._status_ok(msg + "(contracción: probable convergencia)")
        else:
            if solo_warn:
                self._status_err(msg + "≥ 1 (puede no converger)")
            else:
                messagebox.showwarning("Advertencia de convergencia", msg + ": puede divergir.")

    # ---------- Estado ----------
    def _status_ok(self, msg: str):
        self.lbl_estado.configure(text=msg)

    def _status_err(self, msg: str):
        self.lbl_estado.configure(text=msg)

# ==========================
# Main
# ==========================

def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    SimuladorRaices(root)
    root.mainloop()

if __name__ == "__main__":
    main()
