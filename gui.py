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
    derivada_diferencias_finitas
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
        self.zoom_out = tk.BooleanVar(value=True)  # True = zoomed out, False = zoomed in

        self.historia_actual: List[Tuple] = []
        self.historia_comparacion: Dict[str, List[Tuple]] = {}
        self.ultimo_resultado: Optional[float] = None

        # Method configuration: {method_name: {field_name: (required, label, default_value)}}
        self.method_config = {
            "Newton-Raphson": {
                "f": (True, "f(x) *", "x**2 - 2"),
                "df": (False, "f'(x)", ""),
                "x0": (True, "x0 *", "1.5"),
                "tol": (True, "tol *", "1e-8"),
                "max_iter": (True, "max iter *", "50")
            },
            "Secante": {
                "f": (True, "f(x) *", "x**2 - 2"),
                "x0": (True, "x0 *", "1.5"),
                "x1": (True, "x1 *", "2.0"),
                "tol": (True, "tol *", "1e-8"),
                "max_iter": (True, "max iter *", "50")
            },
            "Bisección": {
                "f": (True, "f(x) *", "x**2 - 2"),
                "a": (True, "a *", "0.0"),
                "b": (True, "b *", "2.0"),
                "tol": (True, "tol *", "1e-8"),
                "max_iter": (True, "max iter *", "50")
            },
            "Punto Fijo": {
                "g": (True, "g(x) *", "(x + 2/x)/2"),
                "x0": (True, "x0 *", "1.5"),
                "tol": (True, "tol *", "1e-8"),
                "max_iter": (True, "max iter *", "50")
            },
            "Punto Fijo + Aitken": {
                "f": (True, "f(x) *", "x**2 - 2"),
                "g": (True, "g(x) *", "(x + 2/x)/2"),
                "x0": (True, "x0 *", "1.5"),
                "tol": (True, "tol *", "1e-8"),
                "max_iter": (True, "max iter *", "50")
            },
            "Derivada (Diferencias Finitas)": {
                "f": (True, "f(x) *", "sin(x)"),
                "x": (True, "x *", "1.0"),
                "h": (True, "h *", "1e-4"),
                "esquema": (True, "esquema [progresiva|regresiva|central] *", "central"),
                "d": (True, "orden derivada [1|2] *", "1"),
            },
            "Simpson 1/3": {
                "f": (True, "f(x) *", "sin(x)"),
                "a": (True, "a *", "0"),
                "b": (True, "b *", "3.14159"),
                "n": (True, "n *", "11")
            },
            "Simpson 3/8": {
                "f": (True, "f(x) *", "sin(x)"),
                "a": (True, "a *", "0"),
                "b": (True, "b *", "3.14159"),
                "n": (True, "n *", "11")
            },
            "Boole": {
                "f": (True, "f(x) *", "sin(x)"),
                "a": (True, "a *", "0"),
                "b": (True, "b *", "3.14159"),
                "n": (True, "n *", "11")
            },
            "Trapecio": {
                "f": (True, "f(x) *", "sin(x)"),
                "a": (True, "a *", "0"),
                "b": (True, "b *", "3.14159"),
                "n": (True, "n *", "11")
            },
            "Rectángulo Medio": {
                "f": (True, "f(x) *", "sin(x)"),
                "a": (True, "a *", "0"),
                "b": (True, "b *", "3.14159"),
                "n": (True, "n *", "11")
            }
        }

        # Input field variables
        self.input_vars = {}
        self.input_widgets = {}
        self.input_labels = {}

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
                "Derivada - Diferencias Finitas",
                "Simpson 1/3",
                "Simpson 3/8",
                "Boole",
                "Trapecio",
                "Rectángulo Medio",
            ],
        )
        self.metodo_cb.pack(side=tk.LEFT, padx=6)
        self.metodo_cb.bind(
            "<<ComboboxSelected>>", lambda e: self._update_method_inputs()
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

        # Main content frame with input fields and result display
        main_content = ttk.Frame(tab_fun)
        main_content.pack(fill=tk.X, pady=6)
        
        # Left side: Dynamic input fields container
        self.input_container = ttk.Frame(main_content)
        self.input_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right side: Result display area (initially hidden)
        self.result_frame = ttk.LabelFrame(main_content, text="Resultado", padding=10)
        # Don't pack it initially - it will be shown when there's a result
        
        # Result display label
        self.result_label = ttk.Label(
            self.result_frame, 
            text="", 
            font=("Arial", 14, "bold"),
            foreground="blue"
        )
        self.result_label.pack(pady=10)
        
        # Method info label
        self.method_info_label = ttk.Label(
            self.result_frame, 
            text="", 
            font=("Arial", 10),
            foreground="gray"
        )
        self.method_info_label.pack(pady=5)

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
        
        # Zoom control
        ttk.Checkbutton(
            btns, text="Vista ampliada", variable=self.zoom_out,
            command=self._on_zoom_changed
        ).pack(side=tk.LEFT, padx=12)

        # Graphics display in main tab
        if FigureCanvasTkAgg and plt:
            self.fig, self.ax = plt.subplots(figsize=(7.5, 4.5))
            self.canvas = FigureCanvasTkAgg(self.fig, master=tab_fun)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=6)
        else:
            self.ax = None
            self.canvas = None
            ttk.Label(tab_fun, text="matplotlib no disponible").pack(pady=12)

        tab_tabla = ttk.Frame(nb)
        nb.add(tab_tabla, text="Tabla de iteraciones")

        self.tree = ttk.Treeview(tab_tabla, show="headings", height=18)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Initialize input fields for the default method (after tree is created)
        self._update_method_inputs()

        tab_help = ttk.Frame(nb)
        nb.add(tab_help, text="Ayuda / Teoría")
        help_txt = tk.Text(tab_help, wrap="word", height=20)
        help_txt.pack(fill=tk.BOTH, expand=True)
        help_txt.insert(tk.END, self._texto_ayuda())
        help_txt.configure(state=tk.DISABLED)

        self.lbl_estado = ttk.Label(cont, text="Listo")
        self.lbl_estado.pack(fill=tk.X, pady=(4, 0))


    def _texto_ayuda(self) -> str:
        return (
            "Métodos disponibles:\n\n"
            "RAÍCES:\n"
            "• Bisección: requiere [a,b] con cambio de signo. Convergencia lineal.\n"
            "• Newton-Raphson: requiere f(x) y opcional f'(x). Convergencia cuadrática cerca de la raíz.\n"
            "• Secante: no requiere derivada. Convergencia superlineal.\n"
            "• Punto Fijo: requiere g(x). Converge si |g'(x*)|<1 (contracción).\n"
            "• Aitken: acelera la convergencia del punto fijo.\n\n"
            "DERIVADAS:\n"
            "• Diferencias Finitas: 1ª o 2ª derivada con esquemas progresivo, regresivo o central.\n\n"
            "INTEGRACIÓN:\n"
            "• Simpson 1/3: regla compuesta de Simpson 1/3 para integración numérica.\n"
            "• Simpson 3/8: regla compuesta de Simpson 3/8 para integración numérica.\n"
            "• Boole: regla de Boole para integración numérica de alta precisión.\n"
            "• Trapecio: regla del trapecio compuesta para integración numérica.\n"
            "• Rectángulo Medio: regla del rectángulo medio para integración numérica.\n\n"
            "Sugerir g(x): propone formas g(x)=x-λ f(x) con λ heurístico.\n"
            "Exportación: guarda tabla (CSV), gráfica (PNG) y animación (MP4).\n"
        )

    def _update_method_inputs(self):
        """Update input fields based on selected method."""
        # Clear existing input fields
        for widget in self.input_container.winfo_children():
            widget.destroy()
        
        # Clear input tracking dictionaries
        self.input_vars.clear()
        self.input_widgets.clear()
        self.input_labels.clear()
        
        method = self.current_method.get()
        if method not in self.method_config:
            return
            
        config = self.method_config[method]
        row = 0
        
        # Create grid for input fields
        grid = ttk.Frame(self.input_container)
        grid.pack(fill=tk.X)
        
        for field_name, (required, label, default_value) in config.items():
            # Create label
            label_widget = ttk.Label(grid, text=label)
            label_widget.grid(row=row, column=0, sticky="w", padx=(0, 10))
            self.input_labels[field_name] = label_widget
            
            # Create variable and entry
            var = tk.StringVar(value=default_value)
            self.input_vars[field_name] = var
            
            if field_name in ["f", "df", "g"]:  # Function expressions get wider entries
                entry = ttk.Entry(grid, textvariable=var, width=50)
                entry.grid(row=row, column=1, columnspan=4, sticky="we")
            else:  # Numeric inputs get smaller entries
                entry = ttk.Entry(grid, textvariable=var, width=10)
                entry.grid(row=row, column=1, sticky="w")
            
            self.input_widgets[field_name] = entry
            row += 1
        
        # Update table headers
        self._update_table_headers()

    def _validate_required_inputs(self) -> List[str]:
        """Validate that all required inputs are provided. Returns list of missing fields."""
        method = self.current_method.get()
        if method not in self.method_config:
            return []
        
        missing_fields = []
        config = self.method_config[method]
        
        for field_name, (required, label, _) in config.items():
            if required and field_name in self.input_vars:
                value = self.input_vars[field_name].get().strip()
                if not value:
                    # Remove asterisk from label for error message
                    clean_label = label.replace(" *", "")
                    missing_fields.append(clean_label)
        
        return missing_fields

    def _calculate_zoom_range(self, x0, hist=None, xs_plot=None):
        """Calculate appropriate zoom range based on zoom setting."""
        if self.zoom_out.get():  # Zoomed out view (current behavior)
            if not hist:
                xmin, xmax = x0 - 10, x0 + 10
            else:
                if xs_plot:
                    center = (min(xs_plot) + max(xs_plot)) / 2
                    span = max(xs_plot) - min(xs_plot)
                    span = max(span, 4)  # Minimum span of 4
                    xmin, xmax = center - span * 1.5, center + span * 1.5
                else:
                    xmin, xmax = x0 - 10, x0 + 10
        else:  # Zoomed in view (previous behavior)
            if not hist:
                xmin, xmax = x0 - 5, x0 + 5
            else:
                if xs_plot:
                    xmin, xmax = min(xs_plot) - 1, max(xs_plot) + 1
                else:
                    xmin, xmax = x0 - 5, x0 + 5
        return xmin, xmax

    def _on_zoom_changed(self):
        """Handle zoom setting change by refreshing the graph."""
        if hasattr(self, 'historia_actual') and self.historia_actual:
            self.graficar()

    def _update_result_display(self, result, method_name, iterations=None):
        """Update the result display area with the calculated result."""
        if result is not None:
            # Show the result frame and update content
            if not self.result_frame.winfo_viewable():
                self.result_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            
            if method_name in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
                self.result_label.config(text=f"{result:.8g}")
                self.method_info_label.config(text=f"Integración: {method_name}")
            else:
                self.result_label.config(text=f"{result:.8g}")
                if iterations is not None:
                    self.method_info_label.config(text=f"Raíz encontrada en {iterations} iteraciones")
                else:
                    self.method_info_label.config(text=f"Raíz encontrada: {method_name}")
        else:
            # Hide the result frame when there's no result
            if self.result_frame.winfo_viewable():
                self.result_frame.pack_forget()

    def _update_table_headers(self):
        method = self.current_method.get()
        if method == "Newton-Raphson":
            cols = ("n", "x_n", "f(x_n)", "f'(x_n)", "err_abs", "err_rel")
        elif method == "Secante":
            cols = ("n", "x_n", "x_{n+1}", "err_abs", "err_rel")
        elif method == "Bisección":
            cols = ("n", "a", "b", "m", "f(m)", "err_abs", "err_rel")
        elif method in ["Punto Fijo", "Punto Fijo + Aitken"]:
            cols = ("n", "x_n", "g(x_n)", "err_abs", "err_rel")
        elif method == "Derivada (Diferencias Finitas)":
            cols = ("nivel", "h", "etiqueta", "evals", "aprox", "err_est")
        else:  # Integration methods
            cols = ("Método", "a", "b", "m", "Resultado")
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
        # Validate required inputs first
        missing_fields = self._validate_required_inputs()
        if missing_fields:
            messagebox.showerror(
                "Campos requeridos faltantes", 
                f"Los siguientes campos son obligatorios: {', '.join(missing_fields)}"
            )
            return

        try:
            metodo = self.current_method.get()
            
            # Get inputs based on method
            f = None
            df = None
            g = None
            x0 = None
            x1 = None
            a = None
            b = None
            tol = None
            max_iter = None
            
            # Parse inputs based on what's available for the current method
            if "f" in self.input_vars:
                f = make_safe_func(self.input_vars["f"].get())
            if "df" in self.input_vars and self.input_vars["df"].get().strip():
                df = make_safe_func(self.input_vars["df"].get())
            if "g" in self.input_vars:
                g = make_safe_func(self.input_vars["g"].get())
            if "x0" in self.input_vars:
                x0 = float(self.input_vars["x0"].get())
            if "x1" in self.input_vars:
                x1 = float(self.input_vars["x1"].get())
            if "a" in self.input_vars:
                a = float(self.input_vars["a"].get())
            if "b" in self.input_vars:
                b = float(self.input_vars["b"].get())
            if "tol" in self.input_vars:
                tol = float(self.input_vars["tol"].get())
            if "max_iter" in self.input_vars:
                max_iter = int(self.input_vars["max_iter"].get())
            if "x" in self.input_vars:
                x = float(self.input_vars["x"].get())
            if "h" in self.input_vars:
                h = float(self.input_vars["h"].get())
            if "esquema" in self.input_vars:
                esquema = self.input_vars["esquema"].get().strip().lower()
            if "d" in self.input_vars:
                d_ord = int(self.input_vars["d"].get())
                
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
            elif metodo == "Punto Fijo + Aitken":
                self._verificar_contraccion(g, x0)
                root, hist = punto_fijo_aitken(f, g, x0, tol, max_iter)
            else:  # Integration methods
                n = int(self.input_vars["n"].get()) if "n" in self.input_vars else 11
                result, plan = integrar_nc_compuesto(f, a, b, n, metodo=metodo)
                # Convert integration result to history format for consistency
                hist = [(tag, ai, bi, m, result) for tag, ai, bi, m in plan]
                root = result
        except Exception as e:
            messagebox.showerror("Error en ejecución", str(e))
            return

        self.historia_actual = hist
        self.ultimo_resultado = root
        self._llenar_tabla(hist)

        # Update result display
        iterations = len(hist) if hist else None
        self._update_result_display(root, metodo, iterations)

        if root is not None:
            if metodo == "Derivada (Diferencias Finitas)":
                self._status_ok(f"Derivada ≈ {root:.6g}")
            elif metodo in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
                self._status_ok(f"Integración completada: {root:.6g}")
            else:
                self._status_ok(f"Convergió a {root:.6g}")
        else:
            self._status_err("No convergió con los parámetros dados")
        self.graficar()

    def comparar_metodos(self):
        # Check if current method is an integration method
        method = self.current_method.get()
        if method in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
            messagebox.showinfo(
                "Información", 
                "La comparación de métodos no está disponible para métodos de integración. "
                "Use diferentes métodos de integración individualmente para comparar resultados."
            )
            return
            
        # Validate required inputs first
        missing_fields = self._validate_required_inputs()
        if missing_fields:
            messagebox.showerror(
                "Campos requeridos faltantes", 
                f"Los siguientes campos son obligatorios: {', '.join(missing_fields)}"
            )
            return

        try:
            # Get inputs based on current method configuration
            f = None
            df = None
            g = None
            x0 = None
            x1 = None
            tol = None
            max_iter = None
            
            if "f" in self.input_vars:
                f = make_safe_func(self.input_vars["f"].get())
            if "df" in self.input_vars and self.input_vars["df"].get().strip():
                df = make_safe_func(self.input_vars["df"].get())
            if "g" in self.input_vars:
                g = make_safe_func(self.input_vars["g"].get())
            if "x0" in self.input_vars:
                x0 = float(self.input_vars["x0"].get())
            if "x1" in self.input_vars:
                x1 = float(self.input_vars["x1"].get())
            if "tol" in self.input_vars:
                tol = float(self.input_vars["tol"].get())
            if "max_iter" in self.input_vars:
                max_iter = int(self.input_vars["max_iter"].get())
                
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
            ra, ha = punto_fijo_aitken(f, g, x0, tol, max_iter)
            resultados["Aitken"] = ra
            historias["Aitken"] = ha
        except Exception as e:
            self._status_err(f"Aitken falló: {e}")

        self.historia_comparacion = historias
        self._mostrar_resumen_comparacion(resultados, historias)
        self._grafica_comparacion(f, g, historias)
        
        # Update result display with comparison summary
        if resultados:
            # Show the first successful result
            for method_name, result in resultados.items():
                if result is not None:
                    iterations = len(historias.get(method_name, []))
                    self._update_result_display(result, method_name, iterations)
                    break
            else:
                # No successful results found
                self._update_result_display(None, "Comparación")
        else:
            self._update_result_display(None, "Comparación")
            
        self.nb.select(0)  # Switch to main tab (first tab)

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
            # Try to get x0 from input vars
            if "x0" in self.input_vars:
                xs = [float(self.input_vars["x0"].get())]
            else:
                xs = [0.0]
        
        # Calculate zoom range based on setting
        x0_default = float(self.input_vars["x0"].get()) if "x0" in self.input_vars else 0.0
        xmin, xmax = self._calculate_zoom_range(x0_default, None, xs)
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
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title('Comparación de Métodos')
        self.canvas.draw()

    def graficar(self):
        if not (self.ax and self.canvas):
            return
        metodo = self.current_method.get()
        try:
            f = None
            g = None
            x0 = None
            x_der = None
            
            if "f" in self.input_vars:
                f = make_safe_func(self.input_vars["f"].get())
            if "g" in self.input_vars:
                g = make_safe_func(self.input_vars["g"].get())
            if "x0" in self.input_vars:
                x0 = float(self.input_vars["x0"].get())
        except Exception:
            return

        hist = self.historia_actual
        self.ax.clear()

        if metodo == "Derivada - Diferencias Finitas":
        try:
            x_centro = x_der if x_der is not None else 0.0
            xmin, xmax = self._calculate_zoom_range(x_centro, hist, [x_centro])
            X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]
            Y = [f(x) for x in X]
            self.ax.plot(X, Y, label="f(x)")
            if self.ultimo_resultado is not None and x_der is not None:
                y0 = f(x_der)
                Yt = [self.ultimo_resultado * (x - x_der) + y0 for x in X]
                self.ax.plot(X, Yt, "--", label="Recta tangente aprox.")
                self.ax.plot([x_der], [y0], "ro", label="x, f(x)")
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
            self.ax.set_xlabel('x'); self.ax.set_ylabel('y')
            self.ax.set_title('Derivada por Diferencias Finitas')
            self.canvas.draw()
        except Exception:
            pass
        return
        
        # Handle integration methods differently
        if metodo in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
            if "a" in self.input_vars and "b" in self.input_vars:
                a = float(self.input_vars["a"].get())
                b = float(self.input_vars["b"].get())
                xmin, xmax = a, b
                X = [xmin + i * (xmax - xmin) / 600 for i in range(601)]
                
                try:
                    Y = [f(x) for x in X]
                    self.ax.plot(X, Y, label="f(x)")
                    self.ax.axhline(0, color="k", ls="--")
                    
                    # Shade integration area
                    if hist:
                        for rec in hist:
                            if len(rec) >= 4:
                                tag, ai, bi, m = rec[:4]
                                self.ax.axvspan(ai, bi, alpha=0.12, label=f"Bloque {tag}")
                    
                    self.ax.legend()
                    self.canvas.draw()
                except Exception:
                    pass
            return
        
        # Calculate zoom range based on setting
        xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))] if hist else []
        xmin, xmax = self._calculate_zoom_range(x0, hist, xs_plot)
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
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(f'Gráfica - {metodo}')
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
        
        # Check if current method is an integration method
        if metodo in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
            messagebox.showinfo(
                "Información", 
                "La animación no está disponible para métodos de integración. "
                "Use la función de graficar para visualizar el resultado."
            )
            return
        try:
            f = None
            g = None
            
            if "f" in self.input_vars:
                f = make_safe_func(self.input_vars["f"].get())
            if "g" in self.input_vars:
                g = make_safe_func(self.input_vars["g"].get())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.ax.clear()
        xs_plot = [rec[1] for rec in hist if isinstance(rec[1], (int, float))]
        x0_default = float(self.input_vars["x0"].get()) if "x0" in self.input_vars else 0.0
        xmin, xmax = self._calculate_zoom_range(x0_default, hist, xs_plot)
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
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(f'Animación - {metodo}')

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
        self.nb.select(0)  # Switch to main tab (first tab)

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
        # Clear and hide result display
        if self.result_frame.winfo_viewable():
            self.result_frame.pack_forget()
        self.result_label.config(text="")
        self.method_info_label.config(text="")
        self._status_ok("Limpio")

    def sugerir_g(self):
        # Check if current method is an integration method
        method = self.current_method.get()
        if method in ["Simpson 1/3", "Simpson 3/8", "Boole", "Trapecio", "Rectángulo Medio"]:
            messagebox.showinfo(
                "Información", 
                "La sugerencia de g(x) solo está disponible para métodos de punto fijo."
            )
            return
        
            
        try:
            if "x0" not in self.input_vars or "f" not in self.input_vars:
                messagebox.showerror("Error", "Este método requiere f(x) y x0")
                return
            x0 = float(self.input_vars["x0"].get())
            cands = sugerir_g_desde_f(self.input_vars["f"].get(), x0, make_safe_func)
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
            if sel and "g" in self.input_vars:
                self.input_vars["g"].set(lb.get(sel[0]))
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
                    "Advertencia de convergencia", msg + ": puede diverger."
                )

    def _status_ok(self, msg: str):
        self.lbl_estado.configure(text=msg)

    def _status_err(self, msg: str):
        self.lbl_estado.configure(text=msg)
