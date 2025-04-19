import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
from costs.equipment_cost import pipe_cost, pump_cost
from costs.operational_cost import anual_factor
from costs.tools import friction_factor
from typing import Final
from io import BytesIO
import base64
import flet as ft

# Constants
VISCOSITY: Final[float] = 0.01  # kg/m*s
VOLUMETRIC_FLOW: Final[float] = 50_000  # barrels/day
PIPE_LENGTH: Final[float] = 100  # miles
DENSITY: Final[int] = 930  # kg/m^3
PUMP_EFFICIENCY: Final[float] = 0.85
miles_to_meters: Final[float] = 1609.34
rate: Final[float] = 0.09
nper: Final[int] = 15  # years  

# Standard commercial diameters
STANDARD_DIAMETERS = [0.10226,
                     0.12819, 0.15405, 0.20272, 0.25451, 0.30323,
                     0.33336, 0.381]

def calculate_eaoc(diameter: float) -> tuple[float, float, float]:
    # Pass to SI units
    volumetric_flow = VOLUMETRIC_FLOW * 0.00000184
    pipe_length = PIPE_LENGTH * miles_to_meters
    pipe_length_in_ft = pipe_length / 0.3048
    diameter_in_meters = diameter
    diameter_in_inch = diameter_in_meters / 0.0254  # Corrected conversion from m to inch
    
    reynolds_number = 4 * volumetric_flow * DENSITY / (np.pi * diameter_in_meters * VISCOSITY)
    f_factor = friction_factor(reynolds_number, diameter_in_meters)
    PCpipe = pipe_cost(diameter_in_inch) * pipe_length_in_ft
    pump_power = 32 * f_factor * DENSITY * volumetric_flow**3 * pipe_length / (PUMP_EFFICIENCY * np.pi**2 * diameter_in_meters**5)
    PCpump = 8000 * (pump_power / 1000)**0.6
    f_anual = anual_factor(rate, nper)
    
    PAC = f_anual * (PCpipe + PCpump)
    POC = 0.05 * (pump_power / 1000) * 8760
    EAOC = PAC + POC
    
    return EAOC, diameter_in_inch, pump_power

def generate_plot() -> tuple[BytesIO, tuple]:
    diameters = STANDARD_DIAMETERS
    eaoc_values = []
    for diameter in diameters:
        eaoc, diam_inch, power = calculate_eaoc(diameter)
        eaoc_values.append((diameter, eaoc, diam_inch, power))
    
    optimal = min(eaoc_values, key=lambda x: x[1])
    
    # Styling
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12
    })
    
    plt.figure(figsize=(7, 5))
    plt.plot(diameters, [item[1] for item in eaoc_values])
    plt.scatter(optimal[0], optimal[1], color='g', label=f'Optimal point: {optimal[0]:.3f} m')
    plt.title('Diameter [m] vs EAOC [$]')
    plt.xlabel('Diameter [m]')
    plt.ylabel('EAOC ($/year)')
    plt.grid(True, alpha=0.6)
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')  # Ensure no extra white space
    plt.close()  # Close the figure to free memory
    buf.seek(0)
    return buf, optimal

def main(page: ft.Page):
    page.title = "Pipe Diameter Optimization"
    page.window.width = 800
    page.window.height = 700
    page.window.bgcolor = "system"
    
    img = ft.Image()
    results = ft.Column()
    
    def on_generate(e):
        try:
            buf, optimal = generate_plot()
            img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            results.controls = [
                ft.Text(f"Optimal diameter: {optimal[0]:.4f} m"),
                ft.Text(f"Optimal EAOC: ${optimal[1]:.2f}"),
                ft.Text(f"Diameter in inch: {optimal[2]:.2f} in"),
                ft.Text(f"Pump power: {optimal[3] / 1000:.2f} kW")
            ]
            page.update()
            
        except Exception as ex:
            snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            page.overlay.append(snack_bar)
            snack_bar.open = True
            page.update()
    
    generate_btn = ft.ElevatedButton("Calculate Optimal Diameter", on_click=on_generate)
    
    title = ft.Text("Pipe Diameter Optimization", size=20, weight="bold")
    subtitle = ft.Text("Using Standard Commercial Diameters", italic=True)
    
    page.add(
        ft.Column([
            title,
            subtitle,
            ft.Row([generate_btn], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([img], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([results], alignment=ft.MainAxisAlignment.CENTER)
        ], 
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20)
    )

if __name__ == "__main__":
    ft.app(target=main)