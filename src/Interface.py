import dearpygui.dearpygui as dpg
import tkinter as tk
from tkinter import ttk

class SimulationUI_tk:
    def __init__(self, root):
        """
        Initialize the UI for simulation control.
        """
        self.root = root
        self.cam_dist = False
        self.closed = False
        
        # Internal state
        self.state = {
            "simulation_running": False,
            "simulation_step": False,
            "trace_enabled": False,
            "text_enabled": False,
            "finished": False,
            "plot_type": "particles",
            "plot_variable": "R",
            "camera distance": 10,
        }

        # Plot options and variables
        self.plot_types = ["particles", "line_plot", "surface_plot", 'heatmap']
        self.plot_variables = {
            "particles": ["R", "V"],
            "line_plot": ["Energy", "Momentum", "sim_time", "distribution_V", "distribution_E"],
            "surface_plot": ["phi", "rho"],
            "heatmap": ["phi", "rho"],
        }
        self.current_plot_variables = self.plot_variables[self.state["plot_type"]]

        # Build the UI
        self.build_ui(root)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # ... rest of initialization code ...

    def on_closing(self):
        """Handle window closing event."""
        # Update state to indicate simulation is finished
        self.state["finished"] = True
        self.state["simulation_running"] = False
        self.closed = True
        # You can perform any other cleanup needed here
        self.root.destroy()

    def toggle_simulation(self):
        """Toggle simulation state."""
        self.state["simulation_running"] = not self.state["simulation_running"]
        label = "Stop Simulation" if self.state["simulation_running"] else "Start Simulation"
        self.sim_button.config(text=label)

    def toggle_trace(self):
        """Toggle trace state."""
        self.state["trace_enabled"] = self.trace_var.get()

    def toggle_text(self):
        """Toggle text state."""
        self.state["text_enabled"] = self.text_var.get()

    def update_plot_type(self, event):
        """Update plot type and available plot variables."""
        selected_type = self.plot_type_selector.get()
        self.state["plot_type"] = selected_type
        self.current_plot_variables = self.plot_variables[selected_type]
        self.plot_var_selector["values"] = self.current_plot_variables
        self.plot_var_selector.set(self.current_plot_variables[0])
        self.state["plot_variable"] = self.current_plot_variables[0]

    def select_plot_variable(self, event):
        """Update the selected plot variable."""
        self.state["plot_variable"] = self.plot_var_selector.get()
        
    def simulation_step(self):
        """Step the simulation."""
        self.state["simulation_step"] = True

    def get_camera_distance(self):
        return self.state["camera distance"]
    
    def set_camera_distance(self, distance):
        self.state["camera distance"] = distance

    def build_ui(self, root):
        """Build the simulation control UI."""
        root.title("Simulation Control UI")
        root.geometry("600x400")

        # Start/Stop Button
        self.sim_button = tk.Button(root, text="Start Simulation", command=self.toggle_simulation)
        self.sim_button.pack(pady=10)

        # Step Button
        self.step_button = tk.Button(root, text="Step Simulation", command=self.simulation_step)
        self.step_button.pack(pady=5)

        # Trace and Text Toggles
        self.trace_var = tk.BooleanVar(value=self.state["trace_enabled"])
        self.text_var = tk.BooleanVar(value=self.state["text_enabled"])

        trace_checkbox = tk.Checkbutton(root, text="Enable Trace", variable=self.trace_var, command=self.toggle_trace)
        trace_checkbox.pack(pady=5)

        text_checkbox = tk.Checkbutton(root, text="Enable Text", variable=self.text_var, command=self.toggle_text)
        text_checkbox.pack(pady=5)

        # Separator
        separator = ttk.Separator(root, orient="horizontal")
        separator.pack(fill="x", pady=10)

        # Plot Type Selection
        tk.Label(root, text="Select Plot Type:").pack()
        self.plot_type_selector = ttk.Combobox(root, values=self.plot_types, state="readonly")
        self.plot_type_selector.set(self.state["plot_type"])
        self.plot_type_selector.bind("<<ComboboxSelected>>", self.update_plot_type)
        self.plot_type_selector.pack(pady=5)

        # Plot Variable Selection
        tk.Label(root, text="Select Plot Variable:").pack()
        self.plot_var_selector = ttk.Combobox(root, values=self.current_plot_variables, state="readonly")
        self.plot_var_selector.set(self.state["plot_variable"])
        self.plot_var_selector.bind("<<ComboboxSelected>>", self.select_plot_variable)
        self.plot_var_selector.pack(pady=5)

        self.cam_dist = tk.Scale(root, from_=1, to=20, orient="horizontal", label="Camera Distance")
        self.cam_dist.set(self.state["camera distance"])
        self.cam_dist.pack(pady=5)

    def get_state(self):
        """Return the current state as a dictionary."""
        return self.state

    def update(self):
        """Update the UI state."""
        self.root.update()


class SimulationUI_imgui:
    def __init__(self):
        """
        Initialize the UI for simulation control.
        Internal state is stored inside the object.
        """
        # Internal state
        self.state = {
            "simulation_running": False,
            "trace_enabled": False,
            "text_enabled": True,
            "plot_type": "particles",
            "plot_variable": "R",
        }

        # Plot options and variables
        self.plot_types = ["particles", "line_plot", "surface_plot", 'heatmap']
        self.plot_variables = {
            "particles": ["R", "V"],
            "line_plot": ["Energy", "Momentum", "sim_time", "distribution_V", "distribution_E"],
            "surface_plot": ["phi", "rho"],
            "heatmap": ["phi", "rho"],
        }
        self.current_plot_variables = self.plot_variables[self.state["plot_type"]]

        # Initialize Dear PyGui
        dpg.create_context()
        self.build_ui()

        # Flag to check if the UI should be running
        self.ui_running = True

    def toggle_simulation(self):
        """Toggle simulation state."""
        self.state["simulation_running"] = not self.state["simulation_running"]
        label = "Stop Simulation" if self.state["simulation_running"] else "Start Simulation"
        dpg.configure_item("sim_button", label=label)
        #print(f"Simulation running: {self.state['simulation_running']}")

    def toggle_trace(self, sender, app_data):
        """Toggle trace state."""
        self.state["trace_enabled"] = app_data
        #print(f"Trace enabled: {self.state['trace_enabled']}")

    def toggle_text(self, sender, app_data):
        """Toggle text state."""
        self.state["text_enabled"] = app_data
        #print(f"Text enabled: {self.state['text_enabled']}")

    def update_plot_type(self, sender, app_data):
        """Update plot type and available plot variables."""
        self.state["plot_type"] = app_data
        self.current_plot_variables = self.plot_variables[app_data]
        dpg.configure_item("plot_var_selector", items=self.current_plot_variables)
        self.state["plot_variable"] = self.current_plot_variables[0]
        #print(f"Selected plot type: {self.state['plot_type']}")
        #print(f"Selected plot variable: {self.state['plot_variable']}")

    def select_plot_variable(self, sender, app_data):
        """Update the selected plot variable."""
        self.state["plot_variable"] = app_data
        #print(f"Selected plot variable: {self.state['plot_variable']}")

    def build_ui(self):
        """Build the simulation control UI."""
        dpg.create_viewport(title="Simulation Control UI", width=600, height=500)

        with dpg.window(label="Simulation Control", width=580, height=480):
            # Start/Stop Button
            dpg.add_button(label="Start Simulation", tag="sim_button", callback=self.toggle_simulation)

            # Trace and Text Toggles
            dpg.add_checkbox(label="Enable Trace", default_value=self.state["trace_enabled"], callback=self.toggle_trace)
            dpg.add_checkbox(label="Enable Text", default_value=self.state["text_enabled"], callback=self.toggle_text)

            dpg.add_separator()

            # Plot Type Selection
            dpg.add_text("Select Plot Type:")
            dpg.add_combo(self.plot_types, label="Plot Type", default_value=self.state["plot_type"],
                          callback=self.update_plot_type)

            # Plot Variable Selection
            dpg.add_text("Select Plot Variable:")
            dpg.add_combo(self.current_plot_variables, label="Plot Variable", tag="plot_var_selector",
                          default_value=self.state["plot_variable"],
                          callback=self.select_plot_variable)

    def get_state(self):
        """Return the current state as a dictionary."""
        return self.state

    def start_ui(self):
        """Start the Dear PyGui viewport."""
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def stop_ui(self):
        """Stop the Dear PyGui viewport."""
        self.ui_running = False

    def update_ui(self):
        """Render a single frame of the UI."""
        if self.ui_running and dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
