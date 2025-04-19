import flet as ft
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import asyncio
from costs.operational_cost import operational_cost, anual_factor
from costs.equipment_cost import heat_exchanger_cost, pump_cost, turbine_cost, well_cost
from costs.tools import log_mean_temperature, exchanger_area
from matplotlib import use
use("Agg")  # Use non-GUI backend

# Constants
vaporization_heat = 1400  # kJ/kg
rate = 0.15
nper = 15
default_energy_demand = {'electricity': 700, 'heat': 500}  # kW

# Functions
def geothermal_temperature(depth: int) -> float:
    """Calculate the temperature in depth function.

    Args:
        depth (int): Depth to give geothermal water in meters.

    Returns:
        (float): Temperature at specific depth.
    """
    return 25 + 0.1 * depth

def mass_water_flowrate(energy_demand, temperatures):
    return round(energy_demand["heat"] / (4.184 * (temperatures["3"] - temperatures["5"])), 3)

def calculate_eaoc(depth: int, energy_demand: dict) -> list[float]:
    """Systematic calculous to give the EAOC for the co-generation exercise (ChE 102 - Spring 2013)

    Args:
        depth (int): Depth to give geothermal water in meters.
        energy_demand (dict): Heat and electric energy demand.

    Returns:
        (list[float]): Interest parameters, EOAC, turbine power and refrigerant mass flow rate.
    """
    geo_temperature = geothermal_temperature(depth)
    temperatures = {
        "1": geo_temperature, "2": geo_temperature, "3": geo_temperature, "4": 50,
        "5": 15, "6": 30, "8": 15, "w1": 10, "w2": 47, "9": 20, "7": 26,
        "cw1": 10, "cw2": 12, "sat1": 25, "sat2": 23
    }

    mass_water_3 = mass_water_flowrate(energy_demand, temperatures)
    mass_water_2 = 40 - mass_water_3

    Q_E102 = mass_water_3 * 4.184 * (temperatures["3"] - temperatures["5"])
    lmtd_2 = log_mean_temperature((temperatures["3"] - temperatures["w2"]), (temperatures["5"] - temperatures["w1"]))
    A_E102 = exchanger_area(Q_E102, lmtd_2)

    Q_E101 = mass_water_2 * 4.184 * (temperatures["2"] - temperatures["4"])
    mass_refrigerant = Q_E101 / (1.44 * (temperatures["sat1"] - temperatures["9"]) + vaporization_heat + 0.81 * (temperatures["6"] - temperatures["sat1"]))
    Q1_E101 = mass_refrigerant * 0.81 * (temperatures["6"] - temperatures["sat1"])
    temperatures["s1"] = temperatures["2"] - Q1_E101 / (mass_water_2 * 4.184)
    Q3_E101 = mass_refrigerant * 1.44 * (temperatures["sat1"] - temperatures["9"]) 
    temperatures["s2"] = Q3_E101 / (mass_water_2 * 4.184) + temperatures["4"]
    turbine_power = (Q_E101 / 3) * (1 - (temperatures["cw1"] + 273.15) / (temperatures["1"] + 273.15))
    Q2_E101 = Q_E101 - Q1_E101 - Q3_E101
    Q_E103 = Q_E101 - turbine_power
    
    lmtd_1 = [
        log_mean_temperature((temperatures["2"] - temperatures["6"]), (temperatures["s1"] - temperatures["sat1"])),
        log_mean_temperature((temperatures["s1"] - temperatures["sat1"]), (temperatures["s2"] - temperatures["sat1"])),
        log_mean_temperature((temperatures["s2"] - temperatures["sat1"]), (temperatures["4"] - temperatures["9"]))
    ]
    
    A_E101 = np.sum(list(map(exchanger_area, [Q1_E101, Q2_E101, Q3_E101], lmtd_1)))
    
    mass_cold_water = Q_E103 / (4.184 * (temperatures["cw2"] - temperatures["cw1"]))
    Q1_E103 = mass_refrigerant * 0.81 * (temperatures["7"] - temperatures["sat2"])
    temperatures["s12"] = temperatures["cw2"] - Q1_E103 / (mass_cold_water * 4.184)
    Q3_E103 = mass_refrigerant * 1.44 * (temperatures["sat2"] - temperatures["8"])
    temperatures["s22"] = Q3_E103 / (mass_cold_water * 4.184) + temperatures["cw1"]
    Q2_E103 = Q_E103 - Q1_E103 - Q3_E103
    
    lmtd_3 = [
        log_mean_temperature((temperatures["7"] - temperatures["cw2"]), (temperatures["sat2"] - temperatures["s12"])),
        log_mean_temperature((temperatures["sat2"] - temperatures["s12"]), (temperatures["sat2"] - temperatures["s22"])),
        log_mean_temperature((temperatures["sat2"] - temperatures["s22"]), (temperatures["8"] - temperatures["cw1"]))
    ]
    
    A_E103 = np.sum(list(map(exchanger_area, [Q1_E103, Q2_E103, Q3_E103], lmtd_3)))

    PC_exchangers = np.sum(list(map(heat_exchanger_cost, [A_E101, A_E102, A_E103])))
    PC_turbine = turbine_cost(turbine_power)
    PC_pump = pump_cost(PC_turbine)
    PC_well = well_cost(depth)

    PC = sum([PC_exchangers, PC_turbine, PC_pump, PC_well])
    PAC = anual_factor(rate, nper) * PC
    POC = operational_cost(energy_demand["electricity"], turbine_power)

    return PAC + POC, turbine_power, mass_refrigerant, geo_temperature

def generate_plot(start_depth: int, end_depth: int, steps: int, energy_demand: dict) -> tuple[BytesIO, tuple[float]]:
    """Generate dynamic graph to visualize the EAOC behavior in depth function.

    Args:
        start_depth (int): Left bound
        end_depth (int): Rigth bound in meters.
        steps (int): Step size.
        energy_demand (dict): Heat and electric energy demand.

    Returns:
        tuple[BytesIO, tuple[float]]: Dynamic graph and optimal EAOC, turbine power and refrigerant mass flow rate.
    """
    depths = np.linspace(start_depth, end_depth, steps)
    eaoc_values = [(depth, *calculate_eaoc(depth, energy_demand)) for depth in depths]
    optimal = min(eaoc_values, key=lambda x: x[1])

    # styling
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12
    })
    
    plt.figure(figsize=(8, 6))
    plt.plot(depths, [eaoc for _, eaoc, _, _ in eaoc_values], color="#2B47B9", marker='o', label='EAOC vs Depth')
    plt.scatter(optimal[0], optimal[1], color='g', label=f'Optimal Point: {optimal[0]:.2f} m')
    plt.title('Depth [m] vs EAOC [$]')
    plt.xlabel('Depth (m)')
    plt.ylabel('EAOC (Currency Unit)')
    plt.grid(True, alpha=0.6)
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf, optimal


def main(page: ft.Page):
    page.title = "Geothermal Energy Cost Analysis"
    page.window.width = 800
    page.window.height = 700
    page.window.bgcolor = "system"

    depth_start = ft.TextField(label="Start Depth (m)", value="500", width=150)
    depth_end = ft.TextField(label="End Depth (m)", value="4500", width=150)
    steps = ft.TextField(label="Steps", value="10000", width=100)
    electricity_demand = ft.TextField(label="Electricity Demand (kW)", value="700", width=150)
    heat_demand = ft.TextField(label="Heat Demand (kW)", value="500", width=150)

    img = ft.Image()
    results = ft.Column()

    async def on_generate(e):
        try:
            start_depth = float(depth_start.value)
            end_depth = float(depth_end.value)
            step_count = int(steps.value)
            energy_demand = {
                "electricity": float(electricity_demand.value),
                "heat": float(heat_demand.value)
            }

            loop = asyncio.get_running_loop()
            buf, optimal = await loop.run_in_executor(None, generate_plot, start_depth, end_depth, step_count, energy_demand)
            img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            results.controls = [
                ft.Text(f"Optimal Depth: {optimal[0]:.2f} m"),
                ft.Text(f"Optimal EAOC: ${optimal[1]:.2f}"),
                ft.Text(f"Turbine Power: {optimal[2]:.2f} kW"),
                ft.Text(f"R-134a Mass Flow Rate: {optimal[3]:.2f} kg/s"),
                ft.Text(f"Optimal Temperature: {optimal[4]:.2f} °C")
            ]

            page.update()

        except Exception as ex:
            snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            page.overlay.append(snack_bar)
            snack_bar.open = True
            page.update()

    generate_btn = ft.ElevatedButton("Generate Graph", on_click=on_generate)

    page.add(
        ft.Row([depth_start, depth_end, steps], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([electricity_demand, heat_demand], alignment=ft.MainAxisAlignment.CENTER),
        ft.Row([generate_btn]),
        ft.Row([img, results], alignment=ft.MainAxisAlignment.CENTER)
    )

ft.app(target=main)