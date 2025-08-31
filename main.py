from tkinter import ttk
import tkinter as tk

from gui import SimuladorRaices


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
