import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
sys.stdout.reconfigure(encoding="utf-8")


class TrescaMaterial:
    def __init__(self, E, nu, sigma_y, H_iso, H_kin, mix_ratio=0.5):
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        self.G = E / (2 * (1 + nu))  # Shear modulus
        self.K = E / (3 * (1 - 2 * nu))  # Bulk modulus
        self.sigma_y = sigma_y  # Yield stress
        self.H_iso = H_iso  # Isotropic hardening modulus
        self.H_kin = H_kin  # Kinematic hardening modulus
        self.mix_ratio = mix_ratio  # Hardening mix ratio
        
        # Internal state variables
        self.alpha = np.zeros(6)  # Kinematic backstress
        self.ep_eq = 0  # Equivalent plastic strain
        self.ep = np.zeros(6)  # Plastic strain tensor

    def deviatoric(self, stress):
        mean_stress = np.mean(stress[:3])
        return stress - np.array([mean_stress, mean_stress, mean_stress, 0, 0, 0])
    
    def yield_function(self, stress):
        s = self.deviatoric(stress)
        principal_stresses = np.array([s[0], s[1], s[2]])
        sigma_max = np.max(principal_stresses)
        sigma_min = np.min(principal_stresses)
        return np.abs(sigma_max - sigma_min) - self.sigma_y
    
    def plastic_correction(self, stress_trial):
        if self.yield_function(stress_trial) <= 0:
            return stress_trial
        
        gamma = 0
        H_eff = (1 - self.mix_ratio) * self.H_iso + self.mix_ratio * self.H_kin
        gamma_max = 1e-2
        
        for _ in range(30):
            f = self.yield_function(stress_trial)
            if abs(f) < 1e-6:
                break
            
            dF_dsigma = self.deviatoric(stress_trial)
            norm_dF = np.linalg.norm(dF_dsigma)
            if norm_dF == 0:
                break
            dF_dsigma /= norm_dF
            
            gamma += f / (self.E + H_eff)
            gamma = np.clip(gamma, -gamma_max, gamma_max)
            stress_trial -= gamma * dF_dsigma
        
        self.ep += gamma * dF_dsigma
        self.ep_eq += (1 - self.mix_ratio) * gamma
        self.alpha += self.mix_ratio * gamma * self.H_kin * dF_dsigma
        
        return stress_trial
    
    def update_stress(self, strain_increment, stress_old):
        strain_dev = self.deviatoric(strain_increment)
        stress_trial = stress_old + 2 * self.G * strain_dev + self.K * np.sum(strain_increment[:3]) * np.array([1, 1, 1, 0, 0, 0])
        return self.plastic_correction(stress_trial)


# Initialize the main application window
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Hardening Model GUI")
root.geometry("800x600")

main_frame = ctk.CTkFrame(root)
main_frame.pack(fill="both", expand=True)

status_bar = ctk.CTkFrame(root, height=40)
status_bar.pack(side="bottom", fill="x")

# Global variables for results
graphical_results_frame = None
simulation_results = None  # Store the generated matplotlib figure

# Dictionary to store tab buttons
tabs = {}
active_tab = None
store_inputs = {}  # or store_inputs = []
user_inputs = {}


def show_error(message):
    """Display an error message in a popup window."""
    error_popup = ctk.CTkToplevel(root)
    error_popup.title("Input Error")
    error_popup.geometry("400x200")
    error_popup.grab_set()

    label = ctk.CTkLabel(error_popup, text=message, font=("Arial", 16), text_color="red", wraplength=350)
    label.pack(pady=20, padx=20)

    btn_close = ctk.CTkButton(error_popup, text="OK", font=("Arial", 14), command=error_popup.destroy)
    btn_close.pack(pady=10)

def show_simulation_complete():
    """Display a popup indicating the simulation is complete."""
    complete_popup = ctk.CTkToplevel(root)
    complete_popup.title("Simulation Complete")
    complete_popup.geometry("400x200")
    complete_popup.grab_set()

    label = ctk.CTkLabel(complete_popup, text="Simulation Completed Successfully!\nGo to 'Graphical Results' tab.", font=("Arial", 18, "bold"))
    label.pack(pady=20, padx=20)

    btn_close = ctk.CTkButton(complete_popup, text="OK", font=("Arial", 14), command=complete_popup.destroy)
    btn_close.pack(pady=10)

def validate_inputs(entries):
    """Validates input values."""
    try:
        values = {prop: float(entry.get().strip()) for prop, entry in entries.items()}
    except ValueError:
        show_error("All inputs must be numeric values. Please enter valid numbers.")
        return None

    if not (0 <= values["Mix Ratio"] <= 1):
        show_error("Mix Ratio must be between 0 and 1.\n0 = Isotropic, 1 = Kinematic, values in between = Mixed Hardening.")
        return None

    return values

def run_simulation(entries):
    """Runs the simulation and updates GUI plots."""
    global simulation_results

    # Store the inputs before running the simulation
    store_inputs(entries)

    # Validate Inputs
    values = validate_inputs(entries)
    if values is None:
        return

    print("Running simulation with inputs:", values)

    
    # Assign values to the material object
    mat = TrescaMaterial(
        E=values["Young's Modulus (E)"],
        nu=values["Poisson's Ratio (ν)"],
        sigma_y=values["Yield Stress (σy)"],
        H_iso=values["Isotropic Hardening Modulus (H_iso)"],
        H_kin=values["Kinematic Hardening Modulus (H_kin)"],
        mix_ratio=values["Mix Ratio"]
    )


    # Number of Simulation Steps
    n_steps = 100
    
    strain_increment = np.array([0.0001, 0.00005, 0.00003, 0, 0, 0])
    stress_old = np.zeros(6)
    stress = np.zeros((6, n_steps))
    strain = np.zeros((6, n_steps))

    for step in range(n_steps):
        stress_new = mat.update_stress(strain_increment, stress_old)
        stress[:, step] = stress_new
        strain[:, step] = strain_increment * (step + 1)
        stress_old = stress_new

        # Compute Tresca principal stresses
        deviatoric_stress = mat.deviatoric(stress_new)
        principal_stresses = np.array([deviatoric_stress[0], deviatoric_stress[1], deviatoric_stress[2]])
        sigma_max = np.max(principal_stresses)
        sigma_min = np.min(principal_stresses)
        tresca_stress = np.abs(sigma_max - sigma_min)

        # Compute Maximum Shear Stress (τmax)
        tau_max = (sigma_max - sigma_min) / 2  

    # Compute Allowable Stress (σy / 2N) with N = 1
        factor_of_safety = 1
        allowable_stress = mat.sigma_y / (2 * factor_of_safety)

    # Check for Failure using Tresca's Criterion
        if tau_max < allowable_stress:
            failure_status = "Safe ✅"
        else:
            failure_status = "FAILURE ⚠️"

    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Strain evolution plot
    axs[0, 0].plot(range(n_steps), strain[0, :], 'b', label=r'$\epsilon_{xx}$')
    axs[0, 0].plot(range(n_steps), strain[1, :], 'r', label=r'$\epsilon_{yy}$')
    axs[0, 0].plot(range(n_steps), strain[2, :], 'g', label=r'$\epsilon_{zz}$')
    axs[0, 0].set_title('Strain Evolution')
    axs[0, 0].set_xlabel('Steps')
    axs[0, 0].set_ylabel('Strain')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    mplcursors.cursor(axs[0, 0], hover=True)
    
    # Stress evolution plot
    axs[0, 1].plot(range(n_steps), stress[0, :], 'b', label=r'$\sigma_{xx}$')
    axs[0, 1].plot(range(n_steps), stress[1, :], 'r', label=r'$\sigma_{yy}$')
    axs[0, 1].plot(range(n_steps), stress[2, :], 'g', label=r'$\sigma_{zz}$')
    axs[0, 1].set_title('Stress Evolution')
    axs[0, 1].set_xlabel('Steps')
    axs[0, 1].set_ylabel('Stress')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    mplcursors.cursor(axs[0, 1], hover=True)
    
    # Stress-Strain curve
    axs[1, 0].plot(strain[0, :], stress[0, :], 'b-', label=r'$\sigma_{xx}$ vs $\epsilon_{xx}$')
    axs[1, 0].plot(strain[1, :], stress[1, :], 'r-', label=r'$\sigma_{yy}$ vs $\epsilon_{yy}$')
    axs[1, 0].plot(strain[2, :], stress[2, :], 'g-', label=r'$\sigma_{zz}$ vs $\epsilon_{zz}$')
    axs[1, 0].set_title('Stress-Strain Curve')
    axs[1, 0].set_xlabel('Strain')
    axs[1, 0].set_ylabel('Stress')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    mplcursors.cursor(axs[1, 0], hover=True)
    
    # Yield Surface Evolution
    scale_factor = 1
    theta = np.linspace(0, 2 * np.pi, 7)
    x_initial = scale_factor * mat.sigma_y * np.cos(theta)
    y_initial = scale_factor * mat.sigma_y * np.sin(theta)

    axs[1, 1].plot(x_initial, y_initial, 'k--', linewidth=1, label='Initial Yield Surface')
    
    if mat.mix_ratio == 0:
        x_expanded = (mat.sigma_y + mat.H_iso * mat.ep_eq) * np.cos(theta)
        y_expanded = (mat.sigma_y + mat.H_iso * mat.ep_eq) * np.sin(theta)
        axs[1, 1].plot(x_expanded, y_expanded, 'g-', linewidth=2.5, label='Isotropic Hardening Surface')
    elif mat.mix_ratio == 1:
        translation = scale_factor * (mat.alpha[:2] / mat.sigma_y)
        x_kinematic = x_initial + translation[0]
        y_kinematic = y_initial + translation[1]
        axs[1, 1].plot(x_kinematic, y_kinematic, 'r-', linewidth=2.5, label='Kinematic Hardening Surface')
    else:
        # Isotropic Hardening Surface
        x_expanded = (mat.sigma_y + (1 - mat.mix_ratio) * mat.H_iso * mat.ep_eq) * np.cos(theta)
        y_expanded = (mat.sigma_y + (1 - mat.mix_ratio) * mat.H_iso * mat.ep_eq) * np.sin(theta)
        axs[1, 1].plot(x_expanded, y_expanded, 'g-', linewidth=2.5, label='Isotropic Hardening Surface')
        
        # Kinematic Hardening Surface
        translation = scale_factor * (mat.alpha[:2] / mat.sigma_y)
        x_kinematic = x_initial + translation[0]
        y_kinematic = y_initial + translation[1]
        axs[1, 1].plot(x_kinematic, y_kinematic, 'r-', linewidth=2.5, label='Kinematic Hardening Surface')
        
        # Mixed Hardening Surface (Properly Combined Expansion and Translation)
        mixed_translation = mat.mix_ratio * translation  # Ensuring translation is correctly applied
        x_mixed = (x_expanded + mixed_translation[0])  # Apply translation after expansion
        y_mixed = (y_expanded + mixed_translation[1])
        axs[1, 1].plot(x_mixed, y_mixed, 'b-', linewidth=2.5, label='Mixed Hardening Surface')
    
    axs[1, 1].set_xlabel(r'$\sigma_x - \sigma_y$', fontsize=12)
    axs[1, 1].set_ylabel(r'$\tau_{xy}$', fontsize=12)
    axs[1, 1].set_title('Yield Surface Evolution')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    mplcursors.cursor(axs[1, 1], hover=True)
    
    fig.tight_layout()

    # Store results
    simulation_results = fig

    # Show simulation completion message
    show_simulation_complete()


def clear_inputs(entries):
    """Clear the inputs and any stored results."""
    global user_inputs, simulation_results, graphical_results_frame

    # Clear input fields
    for entry in entries.values():
        entry.delete(0, "end")

    # Permanently clear the user_inputs dictionary
    user_inputs = {}

    # Clear stored results (graphical and numerical)
    simulation_results = None

    # Only attempt to destroy graphical_results_frame if it exists
    if graphical_results_frame is not None:
        graphical_results_frame.destroy()
    
    # Update status bar or show a message to indicate inputs are cleared (optional)
    print("Inputs and results have been cleared.")
    show_error("Inputs and results cleared! Please run a new simulation.")

def load_inputs_tab():
    """Load content for the Inputs tab."""
    outer_frame = ctk.CTkFrame(main_frame)
    outer_frame.pack(expand=True)

    input_frame = ctk.CTkFrame(outer_frame)
    input_frame.pack(pady=20, padx=20, fill="both", expand=True)

    properties = ["Young's Modulus (E)", "Poisson's Ratio (ν)", "Yield Stress (σy)",
                  "Isotropic Hardening Modulus (H_iso)", "Kinematic Hardening Modulus (H_kin)", "Mix Ratio"]
    
    global user_inputs
    entries = {}

    # Load the inputs from the global user_inputs dictionary
    for i, prop in enumerate(properties):
        label = ctk.CTkLabel(input_frame, text=prop, font=("Arial", 18, "bold"))
        label.grid(row=i, column=0, padx=10, pady=8, sticky="e")

        entry = ctk.CTkEntry(input_frame, font=("Arial", 16), width=350)
        entry.grid(row=i, column=1, padx=10, pady=8, sticky="ew")
        entries[prop] = entry

        # Set entry value if already exists in the dictionary (i.e., when switching tabs)
        if prop in user_inputs:  
            entry.insert(0, user_inputs[prop])

    button_frame = ctk.CTkFrame(input_frame)
    button_frame.grid(row=len(properties), column=0, columnspan=2, pady=15)

    btn_run = ctk.CTkButton(button_frame, text="Run Simulation", font=("Arial", 18, "bold"),
                            command=lambda: run_simulation(entries), width=200, height=50)
    btn_run.pack(side="left", padx=20, pady=5)

    btn_clear = ctk.CTkButton(button_frame, text="Clear Inputs", font=("Arial", 18, "bold"),
                              command=lambda: clear_inputs(entries),
                              width=200, height=50)
    btn_clear.pack(side="left", padx=20, pady=5)


def display_graphical_results():
    """Displays the stored graphical results in the 'Graphical Results' tab while keeping the navigation intact."""
    global graphical_results_frame, simulation_results

    # Clear only the main frame content, not the status bar
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Ensure main_frame does not shrink or expand uncontrollably
    main_frame.pack_propagate(False)

    # Create a container frame for the graphical results
    graphical_results_frame = ctk.CTkFrame(main_frame, height=450)  # Adjust height to leave space for tabs
    graphical_results_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Ensure the status bar remains in place
    status_bar.pack(side="bottom", fill="x")

    # Display simulation results if available
    if simulation_results:
        canvas = FigureCanvasTkAgg(simulation_results, master=graphical_results_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    else:
        label = ctk.CTkLabel(graphical_results_frame, text="Run a simulation first to view results.",
                             font=("Arial", 18, "bold"))
        label.pack(pady=20)

    # Force a UI update to ensure elements are correctly positioned
    root.update_idletasks()

        
# Update the store_inputs function to store inputs permanently
def store_inputs(entries):
    """Store inputs in the global dictionary."""
    global user_inputs
    user_inputs = {key: entry.get() for key, entry in entries.items()}

# Add theme toggle button to switch between light and dark themes
def toggle_theme():
    current_theme = ctk.get_appearance_mode()
    new_theme = "Dark" if current_theme == "Light" else "Light"
    ctk.set_appearance_mode(new_theme)

theme_toggle_button = ctk.CTkButton(status_bar, text="Toggle Theme", font=("Arial", 16), command=toggle_theme)
theme_toggle_button.pack(side="right", padx=10, pady=5)


def switch_tab(tab_name):
    """Switch between tabs and highlight the active tab."""
    global active_tab

    # Clear previous tab content
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Load the selected tab's content
    if tab_name == "Inputs":
        load_inputs_tab()
    elif tab_name == "Graphical Results":
        display_graphical_results()

    # Update button colors to reflect the active tab
    for tab in tabs:
        if tab == tab_name:
            tabs[tab].configure(fg_color=("dodger blue", "gray30"), text_color="white")  # Highlighted
        else:
            tabs[tab].configure(fg_color=("gray70", "gray30"), text_color="black")  # Normal

    active_tab = tab_name  # Update active tab

# Create tab buttons
for tab_name in ["Inputs", "Graphical Results", "Numerical Results", "Help"]:
    tabs[tab_name] = ctk.CTkButton(status_bar, text=tab_name, font=("Arial", 16),
                                   command=lambda t=tab_name: switch_tab(t),
                                   fg_color=("gray70", "gray30"))
    tabs[tab_name].pack(side="left", padx=10, pady=5)

# Load the default tab
switch_tab("Inputs")

root.mainloop()
