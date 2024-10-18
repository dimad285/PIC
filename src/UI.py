import tkinter as tk

def toggle_simulation(button):
    global RUN
    RUN = not RUN
    button.config(text="Stop" if RUN else "Start")

def close_window(window):
    global RUN
    global FINISH
    RUN = False
    FINISH = True
    window.destroy()

def toggle_trace():
    global TRACE
    TRACE = not TRACE


def init_UI(window: tk.Tk, size: tuple):

    window.geometry(f'{size[0]}x{size[1]}')
    window.title("PIC Simulation Control")
    button = tk.Button(window, text="Start", command= lambda: toggle_simulation(button))
    button_trace = tk.Button(window, text="Trace", command=toggle_trace)
    button.pack(pady=20)
    button_trace.pack(pady=20)
    window.protocol("WM_DELETE_WINDOW", close_window)