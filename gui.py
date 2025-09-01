import csv
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable, Optional, List, Tuple, Dict

import numpy as np
from utils import make_safe_func, numerical_derivative, sugerir_g_desde_f
from methods import (
    metodo_biseccion,
    newton_raphson,
    metodo_secante,
    punto_fijo,
    punto_fijo_aitken,
    simpson13_bloque,
    simpson38_bloque,
    boole_bloque,
    integrar_nc_compuesto,
)

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except Exception:  # pragma: no cover
    FigureCanvasTkAgg = None
    plt = None
    FuncAnimation = None


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

        top = ttk.Frame(cont)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Método:").pack(side=tk.LEFT)
        self.metodo_cb = ttk.Combobox(
            top,
            textvariable=self.current_method,
            state="readonly",
            values=[
                "Newton-Raphson",
                "Secante",
                "Bisección",
                "Punto Fijo",
                "Punto Fijo + Aitken",
            ],
        )
        self.metodo_cb.pack(side=tk.LEFT, padx=6)
        self.metodo_cb.bind(
            "<<ComboboxSelected>>", lambda e: self._update_table_headers()
        )

        ttk.Label(top, text="Decimales:").pack(side=tk.LEFT, padx=(12, 0))
        tk.Spinbox(top, from_=2, to=15, textvariable=self.decimals, width=5).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Checkbutton(
            top, text="Modo estudiante (explicaciones)", variable=self.modo_estudiante
        ).pack(side=tk.LEFT, padx=12)

        nb = ttk.Notebook(cont)
        nb.pack(fill=tk.BOTH, expand=True, pady=6)
        self.nb = nb

        tab_fun = ttk.Frame(nb)
        nb.add(tab_fun, text="Funciones y Parámetros")

        grid = ttk.Frame(tab_fun)
        grid.pack(fill=tk.X, pady=6)

        ttk.Label(grid, text="f(x):").grid(row=0, column=0, sticky="w")
        self.expr_f = tk.StringVar(value="x**2 - 2")
        ttk.Entry(grid, textvariable=self.expr_f, width=50).grid(
            row=0, column=1, columnspan=4, sticky="we"
        )

        ttk.Label(grid, text="f'(x) (opcional):").grid(row=1, column=0, sticky="w")
        self.expr_df = tk.StringVar(value="")
        ttk.Entry(grid, textvariable=self.expr_df, width=50).grid(
            row=1, column=1, columnspan=4, sticky="we"
        )

        ttk.Label(grid, text="g(x) (punto fijo):").grid(row=2, column=0, sticky="w")
        self.expr_g = tk.StringVar(value="(x + 2/x)/2")
        ttk.Entry(grid, textvariable=self.expr_g, width=50).grid(
            row=2, column=1, columnspan=4, sticky="we"
        )

        ttk.Label(grid, text="x0:").grid(row=3, column=0, sticky="w")
        self.var_x0 = tk.StringVar(value="1.5")
        ttk.Entry(grid, textvariable=self.var_x0, width=10).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Label(grid, text="x1 (Secante):").grid(row=3, column=2, sticky="w")
        self.var_x1 = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.var_x1, width=10).grid(
            row=3, column=3, sticky="w"
        )

        ttk.Label(grid, text="Intervalo [a,b] (Bisección):").grid(
            row=4, column=0, sticky="w"
        )
        self.var_a = tk.StringVar(value="0.0")
        self.var_b = tk.StringVar(value="2.0")
        ttk.Entry(grid, textvariable=self.var_a, width=10).grid(
            row=4, column=1, sticky="w"
        )
        ttk.Entry(grid, textvariable=self.var_b, width=10).grid(
            row=4, column=2, sticky="w"
        )

        ttk.Label(grid, text="tol:").grid(row=5, column=0, sticky="w")
        self.var_tol = tk.StringVar(value="1e-8")
        ttk.Entry(grid, textvariable=self.var_tol, width=10).grid(
            row=5, column=1, sticky="w"
        )

        ttk.Label(grid, text="max iter:").grid(row=5, column=2, sticky="w")
        self.var_max = tk.StringVar(value="50")
        ttk.Entry(grid, textvariable=self.var_max, width=10).grid(
            row=5, column=3, sticky="w"
        )

        btns = ttk.Frame(tab_fun)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="Ejecutar", command=self.ejecutar).pack(side=tk.LEFT)
        ttk.Button(btns, text="Comparar métodos", command=self.comparar_metodos).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Graficar", command=self.graficar).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Animar", command=self.animar).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Guardar CSV", command=self.guardar_csv).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Guardar gráfica", command=self.guardar_png).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(
            btns, text="Guardar animación (.mp4)", command=self.guardar_animacion
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Sugerir g(x)", command=self.sugerir_g).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="Limpiar", command=self.limpiar).pack(
            side=tk.LEFT, padx=6
        )

        tab_tabla = ttk.Frame(nb)
        nb.add(tab_tabla, text="Tabla de iteraciones")

        self.tree = ttk.Treeview(tab_tabla, show="headings", height=18)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self._update_table_headers()

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

        tab_help = ttk.Frame(nb)
        nb.add(tab_help, text="Ayuda / Teoría")
        tab_int = ttk.Frame(nb)
        nb.add(tab_int, text="Integración")
        self._build_integracion(tab_int)  # <-- llamada correcta
        help_txt = tk.Text(tab_help, wrap="word", height=20)
        help_txt.pack(fill=tk.BOTH, expand=True)
        help_txt.insert(tk.END, self._texto_ayuda())
        help_txt.configure(state=tk.DISABLED)

        self.lbl_estado = ttk.Label(cont, text="Listo")
        self.lbl_estado.pack(fill=tk.X, pady=(4, 0))

    def _build_integracion(self, parent):
        frm = ttk.Frame(parent, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        # Entradas
        ttk.Label(frm, text="f(x):").grid(row=0, column=0, sticky="w")
        self.expr_f_int = tk.StringVar(value="sin(x)")
        ttk.Entry(frm, textvariable=self.expr_f_int, width=40).grid(
            row=0, column=1, columnspan=3, sticky="we"
        )

        ttk.Label(frm, text="a").grid(row=1, column=0)
        self.var_a_int = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.var_a_int, width=10).grid(row=1, column=1)

        ttk.Label(frm, text="b").grid(row=1, column=2)
        self.var_b_int = tk.StringVar(value="3.14159")
        ttk.Entry(frm, textvariable=self.var_b_int, width=10).grid(row=1, column=3)

        ttk.Label(frm, text="n").grid(row=1, column=4)
        self.var_n_int = tk.StringVar(value="11")
        ttk.Entry(frm, textvariable=self.var_n_int, width=8).grid(row=1, column=5)

        # Combo de método
        ttk.Label(frm, text="Método").grid(row=2, column=0, sticky="w")
        self.metodo_int = tk.StringVar(value="Simpson 1/3")
        ttk.Combobox(
            frm,
            textvariable=self.metodo_int,
            state="readonly",
            values=[
                "Rectángulo Medio",
                "Trapecio",
                "Simpson 1/3",
                "Simpson 3/8",
                "Boole",
            ],
            width=18,
        ).grid(row=2, column=1, sticky="w")

        # Botón integrar + resultado
        ttk.Button(frm, text="Integrar", command=self._run_integracion).grid(
            row=2, column=2, padx=6
        )
        self.lbl_res_int = ttk.Label(frm, text="Resultado: –")
        self.lbl_res_int.grid(row=2, column=3, columnspan=3, sticky="w")

        # Tabla de bloques usados
        cols = ("Método", "a", "b", "m")
        self.tree_int = ttk.Treeview(frm, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree_int.heading(c, text=c)
            self.tree_int.column(c, width=120, anchor="center")
        self.tree_int.grid(row=3, column=0, columnspan=6, sticky="nsew", pady=6)
        frm.rowconfigure(3, weight=1)

        # Gráfico
        if FigureCanvasTkAgg and plt:
            self.fig_int, self.ax_int = plt.subplots(figsize=(7, 3.8))
            self.canvas_int = FigureCanvasTkAgg(self.fig_int, master=frm)
            self.canvas_int.get_tk_widget().grid(
                row=4, column=0, columnspan=6, sticky="nsew"
            )
        else:
            self.ax_int = None
            self.canvas_int = None

    def _run_integracion(self):
        try:
            f = make_safe_func(self.expr_f_int.get())
            a = float(self.var_a_int.get())
            b = float(self.var_b_int.get())
            n = int(self.var_n_int.get())
            metodo = self.metodo_int.get()
            I, plan = integrar_nc_compuesto(f, a, b, n, metodo=metodo)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Mostrar resultado
        self.lbl_res_int.config(text=f"Resultado: {I:.10g}")

        # Llenar tabla con el plan de bloques
        for r in self.tree_int.get_children():
            self.tree_int.delete(r)
        for tag, ai, bi, m in plan:
            self.tree_int.insert("", "end", values=(tag, f"{ai:.6g}", f"{bi:.6g}", m))

        # Graficar f(x) y sombrear bloques
        if self.ax_int and self.canvas_int:
            X = np.linspace(a, b, 600)
            Y = [f(x) for x in X]
            self.ax_int.clear()
            self.ax_int.grid(True, ls=":")
            self.ax_int.plot(X, Y, label="f(x)")
            self.ax_int.axhline(0, color="k", ls="--")
            for tag, ai, bi, _ in plan:
                self.ax_int.axvspan(ai, bi, alpha=0.12)  # sombrear bloque
            self.ax_int.legend()
            self.canvas_int.draw()

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
            self.tree.column(c, width=120, anchor="center")

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
            self.tree.insert("", "end", values=tuple(vals))

    def ejecutar(self):
        try:
            metodo = self.current_method.get()
            f = make_safe_func(self.expr_f.get())
            df = (
                make_safe_func(self.expr_df.get())
                if self.expr_df.get().strip()
                else None
            )
            g = make_safe_func(self.expr_g.get())
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
            else:
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
        try:
            f = make_safe_func(self.expr_f.get())
            df = (
                make_safe_func(self.expr_df.get())
                if self.expr_df.get().strip()
                else None
            )
            g = make_safe_func(self.expr_g.get())
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
        self._mostrar_resumen_comparacion(resultados, historias)
        self._grafica_comparacion(f, g, historias)
        self.nb.select(2)

    def _mostrar_resumen_comparacion(self, resultados, historias):
        for row in self.tree.get_children():
            self.tree.delete(row)
        cols = ("Método", "Raíz", "Iteraciones")
        self.tree["columns"] = cols
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=160, anchor="center")
        for m, r in resultados.items():
            iters = len(historias.get(m, []))
            r_txt = "-" if r is None else f"{r:.6g}"
            self.tree.insert("", "end", values=(m, r_txt, iters))

    def _grafica_comparacion(self, f, g, historias):
        if not (self.ax and self.canvas):
            return
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
        try:
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, label="f(x)")
            self.ax.axhline(0, color="k", ls="--")
        except Exception:
            pass

        for metodo, hist in historias.items():
            xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
            if not xs_plot:
                continue
            try:
                self.ax.plot(
                    xs_plot,
                    [f(x) for x in xs_plot],
                    "o-",
                    label=f"Iteraciones {metodo}",
                )
            except Exception:
                pass
        self.ax.legend()
        self.canvas.draw()

    def graficar(self):
        if not (self.ax and self.canvas):
            return
        metodo = self.current_method.get()
        try:
            f = make_safe_func(self.expr_f.get())
            g = make_safe_func(self.expr_g.get())
            x0 = float(self.var_x0.get())
        except Exception:
            return

        hist = self.historia_actual
        self.ax.clear()
        if not hist:
            xmin, xmax = x0 - 5, x0 + 5
        else:
            xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
            xmin, xmax = (
                (min(xs_plot) - 1, max(xs_plot) + 1) if xs_plot else (x0 - 5, x0 + 5)
            )
        X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]

        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            try:
                Yg = [g(x) for x in X]
                self.ax.plot(X, Yg, label="g(x)")
                self.ax.plot(X, X, "--", label="y=x")
                if hist:
                    xs_plot = [
                        rec[1] for rec in hist if isinstance(rec[1], (int, float))
                    ]
                    self.ax.plot(
                        xs_plot, [g(x) for x in xs_plot], "o-", label="Iteraciones"
                    )
                if self.ultimo_resultado is not None:
                    self.ax.plot(
                        self.ultimo_resultado,
                        self.ultimo_resultado,
                        "ro",
                        markersize=8,
                        label="Punto fijo estimado",
                    )
            except Exception:
                pass
        else:
            try:
                Y = [f(x) for x in X]
                self.ax.plot(X, Y, label="f(x)")
                self.ax.axhline(0, color="k", ls="--")
                if hist:
                    xs_plot = [
                        rec[1] for rec in hist if isinstance(rec[1], (int, float))
                    ]
                    self.ax.plot(
                        xs_plot, [f(x) for x in xs_plot], "o-", label="Iteraciones"
                    )
                if self.ultimo_resultado is not None:
                    self.ax.plot(
                        self.ultimo_resultado,
                        0,
                        "ro",
                        markersize=8,
                        label="Raíz estimada",
                    )
            except Exception:
                pass
        self.ax.legend()
        self.canvas.draw()

    def animar(self):
        if not (self.ax and self.canvas and FuncAnimation):
            messagebox.showerror(
                "Error", "Animación no disponible (matplotlib.animation)"
            )
            return
        hist = self.historia_actual
        if not hist:
            messagebox.showwarning(
                "Atención", "Ejecuta primero un método para generar iteraciones"
            )
            return
        metodo = self.current_method.get()
        try:
            f = make_safe_func(self.expr_f.get())
            g = make_safe_func(self.expr_g.get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.ax.clear()
        xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
        xmin, xmax = (min(xs_plot) - 1, max(xs_plot) + 1) if xs_plot else (-5, 5)
        X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]
        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            Yg = [g(x) for x in X]
            self.ax.plot(X, Yg, label="g(x)")
            self.ax.plot(X, X, "--", label="y=x")
        else:
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, label="f(x)")
            self.ax.axhline(0, color="k", ls="--")
        (linea_iter,) = self.ax.plot([], [], "o-", label="Iteraciones")
        self.ax.legend()

        if metodo in ("Punto Fijo", "Punto Fijo + Aitken"):
            ys = [g(x) for x in xs_plot]
        else:
            ys = [f(x) for x in xs_plot]

        def init():
            linea_iter.set_data([], [])
            return (linea_iter,)

        def update(i):
            linea_iter.set_data(xs_plot[: i + 1], ys[: i + 1])
            return (linea_iter,)

        self.anim = FuncAnimation(
            self.fig,
            update,
            init_func=init,
            frames=len(xs_plot),
            interval=600,
            blit=True,
        )
        self.canvas.draw()
        self.nb.select(2)

    def guardar_csv(self):
        if not self.historia_actual and not self.historia_comparacion:
            messagebox.showerror("Error", "No hay datos para guardar")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "w", newline="") as f:
                w = csv.writer(f)
                if self.historia_comparacion:
                    w.writerow(["Resumen de comparación"])
                    w.writerow(["Método", "x", "Iteraciones"])
                    for m, hist in self.historia_comparacion.items():
                        x = None
                        if hist:
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
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")]
        )
        if not file_path:
            return
        try:
            self.fig.savefig(file_path, bbox_inches="tight", dpi=150)
            messagebox.showinfo("Éxito", f"Gráfica guardada en {file_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def guardar_animacion(self):
        if not hasattr(self, "anim") or self.anim is None:
            messagebox.showerror(
                "Error", "Primero crea la animación con el botón 'Animar'"
            )
            return
        if not FuncAnimation:
            messagebox.showerror("Error", "Animación no disponible")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4", filetypes=[("MP4", "*.mp4")]
        )
        if not file_path:
            return
        try:
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
            cands = sugerir_g_desde_f(self.expr_f.get(), x0, make_safe_func)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if not cands:
            messagebox.showwarning(
                "Sugerencias", "No se pudieron generar sugerencias para g(x)"
            )
            return
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

    def _verificar_contraccion(
        self, g: Callable[[float], float], x0: float, solo_warn: bool = False
    ):
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
                messagebox.showwarning(
                    "Advertencia de convergencia", msg + ": puede divergir."
                )

    def _status_ok(self, msg: str):
        self.lbl_estado.configure(text=msg)

    def _status_err(self, msg: str):
        self.lbl_estado.configure(text=msg)
