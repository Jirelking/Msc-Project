import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser


import os
import sys
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

# Prevent DPI scaling errors
os.environ['TK_FORCE_FIXED_DPI'] = '1'

# Check if running as .exe and fix stdout/stderr
if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w', encoding="utf-8")
    sys.stderr = open(os.devnull, 'w', encoding="utf-8")







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
            return stress_trial, self.ep, self.alpha  # No plastic flow, return all unchanged values

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

        # Update plastic strain and internal variables
        self.ep += gamma * dF_dsigma
        self.ep_eq += (1 - self.mix_ratio) * gamma
        self.alpha += self.mix_ratio * gamma * self.H_kin * dF_dsigma

        return stress_trial, self.ep, self.alpha  # Return updated values

    def update_stress(self, strain_increment, stress_old):
        """
        Updates stress and computes plastic strain correction.

        Parameters:
            strain_increment (np.array): Incremental strain tensor.
            stress_old (np.array): Previous stress state.

        Returns:
            tuple: (Updated stress tensor, Updated plastic strain tensor, Updated kinematic backstress tensor)
        """
        # Compute deviatoric strain component
        strain_dev = self.deviatoric(strain_increment)

        # Compute trial stress using elasticity
        stress_trial = (
            stress_old 
            + 2 * self.G * strain_dev 
            + self.K * np.sum(strain_increment[:3]) * np.array([1, 1, 1, 0, 0, 0])
        )

        # Apply plastic correction and retrieve updated values
        stress_corrected, plastic_strain, alpha_new = self.plastic_correction(stress_trial)
    
        # Update kinematic backstress tensor
        self.alpha = alpha_new

        return stress_corrected, plastic_strain, alpha_new


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
inputs_frame = None

# Dictionary to store tab buttons
tabs = {}
active_tab = None
store_inputs = {}  # or store_inputs = []
user_inputs = {}

current_tab = "Inputs"  # Default tab


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

    label = ctk.CTkLabel(complete_popup, text="Simulation Completed Successfully!\nGo to the Results' tabs \nto view the outcome.", font=("Arial", 18, "bold"))
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
    global simulation_results, numerical_results, numerical_results_frame

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
    plastic_strain = np.zeros((6, n_steps))
    kinematic_backstress = np.zeros((6, n_steps))
    
    # Initialize storage for numerical results
    numerical_results = []


    for step in range(n_steps):
        stress[:, step], plastic_strain[:, step], kinematic_backstress[:, step] = mat.update_stress(strain_increment, stress_old)
        strain[:, step] = strain_increment * (step + 1)
        stress_old = stress[:, step]


        # Compute Tresca principal stresses
        deviatoric_stress = mat.deviatoric(stress[:, step])
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

    
    # Store results in a dictionary
        numerical_results.append({
            "Step": step + 1,
            "Total Strain Tensor": strain[:, step],
            "Plastic Strain Tensor": plastic_strain[:, step],
            "Kinematic Backstress Tensor": kinematic_backstress[:, step],
            "Stress Tensor": stress[:, step],
            "Tresca Principal Stresses (Max, Min)": (sigma_max, sigma_min),
            "Maximum Shear Stress": tau_max,
            "Allowable Stress": allowable_stress,
            "Failure Status": failure_status
        })
    

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
    axs[1, 1].set_title('TRESCA Yield Surface Evolution (Hexagonal)')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    mplcursors.cursor(axs[1, 1], hover=True)
    
    fig.tight_layout()

    # Store results
    simulation_results = fig


    # Make sure numerical_results_frame is placed inside the correct Numerical Results tab
    numerical_results_frame = create_numerical_results_frame()

    # Hide it first in case it's placed somewhere else
    numerical_results_frame.pack_forget()

    # Ensure it is only shown inside the Numerical Results tab
    if current_tab == "Numerical Results":  
        numerical_results_frame.pack(fill="both", expand=True)

    

    # Update the Numerical Results tab
    update_numerical_results(numerical_results)

    # Store numerical results but don't update the table yet
    store_numerical_results(numerical_results)

    # Show simulation completion message
    show_simulation_complete()


def clear_inputs(entries):
    """Clear the inputs and any stored results."""
    global user_inputs, simulation_results, graphical_results_frame, numerical_results, numerical_results_frame

    # Clear input fields
    for entry in entries.values():
        entry.delete(0, "end")

    # Permanently clear the user_inputs dictionary
    user_inputs = {}

    # Clear stored results (graphical and numerical)
    simulation_results = None
    numerical_results = []  # Reset numerical results list

    # Clear Graphical Results Frame
    if graphical_results_frame is not None and graphical_results_frame.winfo_exists():
        graphical_results_frame.destroy()

    # Clear Numerical Results Frame (with proper existence check)
    if numerical_results_frame is not None and numerical_results_frame.winfo_exists():
        for widget in numerical_results_frame.winfo_children():
            widget.destroy()  # Clear the contents
        numerical_results_frame.pack_forget()  # Hide the frame to remove blank space

    # Update status bar or show a message to indicate inputs are cleared (optional)
    print("Inputs and results have been cleared.")
    show_error("Inputs and results cleared! Please run a new simulation.")


def load_inputs_tab():
    """Load content for the Inputs tab, ensuring persistence."""
    global inputs_frame, user_inputs

    # Ensure inputs_frame exists and is not recreated unnecessarily
    if inputs_frame is None or not inputs_frame.winfo_exists():
        inputs_frame = ctk.CTkFrame(main_frame)
        inputs_frame.pack(fill="both", expand=True)

        outer_frame = ctk.CTkFrame(inputs_frame)
        outer_frame.pack(expand=True)

        input_frame = ctk.CTkFrame(outer_frame)
        input_frame.pack(pady=20, padx=20, fill="both", expand=True)

        properties = ["Young's Modulus (E)", "Poisson's Ratio (ν)", "Yield Stress (σy)",
                      "Isotropic Hardening Modulus (H_iso)", "Kinematic Hardening Modulus (H_kin)", "Mix Ratio"]
        
        entries = {}

        # Ensure user_inputs dictionary exists
        if "user_inputs" not in globals():
            user_inputs = {}

        # Load the inputs from user_inputs dictionary
        for i, prop in enumerate(properties):
            label = ctk.CTkLabel(input_frame, text=prop, font=("Arial", 18, "bold"))
            label.grid(row=i, column=0, padx=10, pady=8, sticky="e")

            entry = ctk.CTkEntry(input_frame, font=("Arial", 16), width=350)
            entry.grid(row=i, column=1, padx=10, pady=8, sticky="ew")
            entries[prop] = entry

            # Restore previously entered values
            if prop in user_inputs:
                entry.insert(0, user_inputs[prop])  # Insert stored value (not widget)

        button_frame = ctk.CTkFrame(input_frame)
        button_frame.grid(row=len(properties), column=0, columnspan=2, pady=15)

        btn_run = ctk.CTkButton(button_frame, text="Run Simulation", font=("Arial", 18, "bold"),
                                command=lambda: run_simulation(entries), width=200, height=50)
        btn_run.pack(side="left", padx=20, pady=5)

        btn_clear = ctk.CTkButton(button_frame, text="Clear Inputs", font=("Arial", 18, "bold"),
                                  command=lambda: clear_inputs(entries),
                                  width=200, height=50)
        btn_clear.pack(side="left", padx=20, pady=5)

        # Store only the values, not the entry widgets
        user_inputs = {prop: entries[prop].get() for prop in properties}

    else:
        inputs_frame.pack(fill="both", expand=True)  # Ensure it's shown when switching tabs


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


class NumericalResultsTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Define the columns
        columns = ("Step", "Total Strain", "Plastic Strain", "Backstress",
                   "Stress", "Tresca (Max)", "Tresca (Min)", "Max Shear", 
                   "Allowable Stress", "Failure Status")

        # Create a Treeview widget
        self.tree = ttk.Treeview(self, columns=columns, show="headings")

        # Define column headings
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor="center")  # Adjust width as needed

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Layout
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def update_results(self, results_data):
        """Populates the table with simulation results."""
        # Clear existing data
        self.tree.delete(*self.tree.get_children())

        # Insert new data
        for row in results_data:
            self.tree.insert("", "end", values=row)


def update_numerical_results(numerical_results):
    """Update the Numerical Results tab with the simulation data."""
    global numerical_results_frame

    # Ensure the numerical results frame exists and clear previous data
    if numerical_results_frame is None or not numerical_results_frame.winfo_exists():
        numerical_results_frame = ctk.CTkFrame(main_frame)

    for widget in numerical_results_frame.winfo_children():
        widget.destroy()

    # Define table columns
    columns = [
        "Step", "Total Strain", "Plastic Strain", "Kinematic Backstress",
        "Stress", "Tresca Max", "Tresca Min", "Max Shear Stress",
        "Allowable Stress", "Failure Status"
    ]

    # Create the treeview table
    tree = ttk.Treeview(numerical_results_frame, columns=columns, show="headings", height=15)

    # Define column widths (grouping certain columns closer)
    column_widths = {
        "Step": 60,
        "Total Strain": 300,
        "Plastic Strain": 300,
        "Kinematic Backstress": 340,
        "Stress": 320,
        "Tresca Max": 110,  # Close together
        "Tresca Min": 110,  # Close together
        "Max Shear Stress": 150,  # Close together
        "Allowable Stress": 150,  # Close together
        "Failure Status": 130  # Close together
    }

    # Apply column properties
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=column_widths[col], anchor="center", stretch=False)

    # Insert data into the table
    for result in numerical_results:
        tree.insert("", "end", values=(
            result["Step"],
            ', '.join(f"{x:.5f}" for x in result["Total Strain Tensor"]),
            ', '.join(f"{x:.5f}" for x in result["Plastic Strain Tensor"]),
            ', '.join(f"{x:.5f}" for x in result["Kinematic Backstress Tensor"]),
            ', '.join(f"{x:.5f}" for x in result["Stress Tensor"]),
            f"{result['Tresca Principal Stresses (Max, Min)'][0]:.5f}",
            f"{result['Tresca Principal Stresses (Max, Min)'][1]:.5f}",
            f"{result['Maximum Shear Stress']:.5f}",
            f"{result['Allowable Stress']:.5f}",
            result["Failure Status"]
        ))

    # Ensure bold headers
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

    # Pack the treeview widget
    tree.pack(fill="both", expand=True)

    # Ensure the frame is only shown if the user is on the "Numerical Results" tab
    if current_tab == "Numerical Results":
        numerical_results_frame.pack(fill="both", expand=True)
    else:
        numerical_results_frame.pack_forget()  # Hide it when not in use



def create_numerical_results_frame():
    """Creates the numerical results frame inside the Numerical Results tab."""
    global numerical_results_frame

    # Destroy the existing frame if it exists to refresh the content
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Create a new frame for numerical results
    numerical_results_frame = ctk.CTkFrame(main_frame)
    numerical_results_frame.pack(fill="both", expand=True)

    # Create a table for numerical results
    table = ttk.Treeview(numerical_results_frame, columns=(
        "Step", "Total Strain Tensor", "Plastic Strain Tensor", "Kinematic Backstress Tensor", 
        "Stress Tensor", "Tresca Max", "Tresca Min", "Max Shear Stress", 
        "Allowable Stress", "Failure Status"
    ), show="headings")

    # Adjust column widths to space columns properly
    column_widths = {
        "Step": 60, 
        "Total Strain Tensor": 300,  # More space
        "Plastic Strain Tensor": 300,  # More space
        "Kinematic Backstress Tensor": 340,  # More space
        "Stress Tensor": 320,  # More space
        "Tresca Max": 110,  # Close together
        "Tresca Min": 110,  # Close together
        "Max Shear Stress": 150,  # Close together
        "Allowable Stress": 150,  # Close together
        "Failure Status": 130  # Close together
    }

    # Apply column widths and center alignment
    for col, width in column_widths.items():
        table.column(col, width=width, anchor="center")
        table.heading(col, text=col, anchor="center")

    # Apply a bold font to the headers for better distinction
    style = ttk.Style()
    style.configure("Treeview.Heading", font=("Arial", 12, "bold"))

    # Pack the table into the frame
    table.pack(fill="both", expand=True)

    return numerical_results_frame


def store_numerical_results(results):
    global numerical_results
    numerical_results = results

    # Only update the table if the user is currently on the "Numerical Results" tab
    if active_tab == "Numerical Results":
        update_numerical_results(numerical_results)

        
# Update the store_inputs function to store inputs permanently
def store_inputs(entries):
    """Store inputs in the global dictionary."""
    global user_inputs
    user_inputs = {key: entry.get() for key, entry in entries.items()}




# Dummy GitHub link for source code
github_repo_url = "https://github.com/example/repository"

# Function to open user manual
def open_user_manual():
    manual_path = os.path.join(os.getcwd(), "User_Manual.pdf")
    if os.path.exists(manual_path):
        os.startfile(manual_path)  # Open locally
    else:
        webbrowser.open("https://github.com/example/repository/raw/main/User_Manual.pdf")  # Fetch from GitHub

# Function to open GitHub repository
def open_github_repo():
    webbrowser.open(github_repo_url)

# Function to create Help tab content
def load_help_tab():
    global help_frame
    help_frame = ctk.CTkFrame(main_frame)
    help_frame.pack(fill="both", expand=True)

    # Theme-sensitive text colors
    text_color = "dodger blue" if ctk.get_appearance_mode() == "Dark" else "blue"
    hover_color = "cyan" if ctk.get_appearance_mode() == "Dark" else "navy"

    # Detect current theme mode
    current_theme = ctk.get_appearance_mode()  # Returns "Light" or "Dark"

    if current_theme == "Dark":
        text_color = ("light blue", "white")  # Brighter for dark mode
        hover_color = ("cyan", "yellow")  # More visible in dark mode
    else:
        text_color = ("dodger blue", "light blue")
        hover_color = ("navy", "cyan")

    # Centering Frame
    content_frame = ctk.CTkFrame(help_frame, fg_color="transparent")
    content_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Function to open links
    def open_user_manual():
        manual_path = os.path.abspath("User_Manual.pdf")
        webbrowser.open(f"file://{manual_path}")  # Opens local manual

    def open_github_repo():
        webbrowser.open("https://github.com/dummy_repo_link")  # Replace with actual GitHub link

    # Download User Manual label
    manual_label = ctk.CTkLabel(content_frame, text="Find Help: \n 📘 By Downloading the User Manual", 
                                font=("Arial", 18, "bold"), text_color=text_color, cursor="hand2")
    manual_label.pack(pady=15)
    manual_label.bind("<Enter>", lambda e: manual_label.configure(font=("Arial", 18, "bold", "underline"), text_color=hover_color))
    manual_label.bind("<Leave>", lambda e: manual_label.configure(font=("Arial", 18, "bold"), text_color=text_color))
    manual_label.bind("<Button-1>", lambda e: open_user_manual())

    # View Source Code label
    source_label = ctk.CTkLabel(content_frame, text="💻 View Source Code", 
                                font=("Arial", 18, "bold"), text_color=text_color, cursor="hand2")
    source_label.pack(pady=15)
    source_label.bind("<Enter>", lambda e: source_label.configure(font=("Arial", 18, "bold", "underline"), text_color=hover_color))
    source_label.bind("<Leave>", lambda e: source_label.configure(font=("Arial", 18, "bold"), text_color=text_color))
    source_label.bind("<Button-1>", lambda e: open_github_repo())





# Add theme toggle button to switch between light and dark themes
def toggle_theme():
    current_theme = ctk.get_appearance_mode()
    new_theme = "Dark" if current_theme == "Light" else "Light"
    ctk.set_appearance_mode(new_theme)

theme_toggle_button = ctk.CTkButton(status_bar, text="Toggle Theme", font=("Arial", 16), command=toggle_theme)
theme_toggle_button.pack(side="right", padx=10, pady=5)


def switch_tab(tab_name):
    """Switch between tabs and highlight the active tab."""
    global active_tab, numerical_results_frame, inputs_frame, current_tab, numerical_results

    # Ensure frames and variables exist to avoid NameError
    if "numerical_results_frame" not in globals():
        numerical_results_frame = None
    if "inputs_frame" not in globals():
        inputs_frame = None
    if "numerical_results" not in globals():
        numerical_results = []

    # Clear previous tab content
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Update the current tab tracker
    current_tab = tab_name

    # Load the selected tab's content
    if tab_name == "Inputs":
        load_inputs_tab()

    elif tab_name == "Graphical Results":
        display_graphical_results()

    elif tab_name == "Numerical Results":
        if numerical_results_frame is None or not numerical_results_frame.winfo_exists():
            numerical_results_frame = ctk.CTkFrame(main_frame)  # Recreate frame if it doesn't exist
        numerical_results_frame.pack(fill="both", expand=True)

        # Ensure numerical results exist before checking
        if not numerical_results:
            placeholder_label = ctk.CTkLabel(numerical_results_frame, 
                                             text="Run simulation to view numerical results",
                                             font=("Arial", 18, "bold"))
            placeholder_label.pack(pady=20)
        else:
            update_numerical_results(numerical_results)  # Populate the table if results exist

    elif tab_name == "Help":
        load_help_tab()  # Ensure Help content loads properly

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

def on_closing():
    try:
        root.destroy()  # Close the Tkinter application
    except:
        pass  # Ignore any errors

root.protocol("WM_DELETE_WINDOW", on_closing)  # Properly handle closing


root.mainloop()
