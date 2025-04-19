import flet as ft
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import asyncio
from costs.operational_cost import anual_factor
from costs.equipment_cost import heat_exchanger_cost
from costs.tools import log_mean_temperature, exchanger_area
from matplotlib import use
use("Agg")  # Use non-GUI backend


class ThermalConstants:
    """Constants used in thermal calculations."""
    VAPORIZATION_HEAT = 1400  # kJ/kg
    HEAT_CAPACITY = 4.184  # J/g*K
    PRESSURE_IN_5 = 0.0169  # MPa
    ABSORBER_NUMBER = 2
    DEFAULT_TEMPERATURE = 50  # °C


class AbsorberChillingSystem:
    """Class representing the geothermal absorber chilling system."""
    
    def __init__(self, energy_demand=None):
        """Initialize the chilling system with energy demand."""
        self.energy_demand = energy_demand or {"Ice Museum": 500}  # kW
        self.temperatures = {
            "4": 10, "6": -25, "9": -20, "10": -5, 
            "geo_in": 75, "cw_in": 4.5, "5": -25, "cw_out": 20
        }
        self.ammonia_mass = {}
        self.rate = 0.07
        self.nper = 10
        
    @staticmethod
    def pressure(temperature: float) -> float:
        """Calculate pressure as temperature function.
        
        Args:
            temperature (float): Temperature in Celsius degree.
            
        Returns:
            float: pressure in MPa.
        """
        
        return 0.26 * np.exp(0.025 * temperature)
    
    
    @staticmethod
    def water_composition(temperature: float) -> float:
        """Water composition as temperature function.
        
        Args:
            temperature (float): temperature in Celsius degree.
            
        Returns:
            float: adimensional water composition.
        """
        return (1.89 * temperature + 1.41) / 10_000
    
    @staticmethod
    def liquid_mass_ratio(water_composition: float) -> float:
        """m_2/m_1 liquid mass ratio.
        
        Args:
            water_composition (float): Water composition.
            
        Returns:
            float: m_2/m_1 ratio.
        """
        return 0.905 - 8.39 * (water_composition * water_composition) - 0.708 * water_composition
    
    def calculate_eaoc(self, temperature: float) -> list[float]:
        """Calculate EAOC for a given temperature.
        
        Args:
            temperature (float): Operating temperature in Celsius.
            
        Returns:
            list[float]: [EAOC, geo_mass_flow, cold_mass_flow, chilled_mass_flow]
        """
        # Initial calculations
        pressure_in_1 = self.pressure(temperature)
        water_comp = self.water_composition(temperature)
        mass_ratio = self.liquid_mass_ratio(water_comp)
        
        # Mass calculations
        self.ammonia_mass["5"] = self.energy_demand["Ice Museum"] / ThermalConstants.VAPORIZATION_HEAT
        self.ammonia_mass["1"] = self.ammonia_mass["5"] / (1 - mass_ratio)
        self.ammonia_mass["2"] = self.ammonia_mass["1"] - self.ammonia_mass["5"]
        self.temperatures["8"] = (self.ammonia_mass["2"] * temperature + self.ammonia_mass["5"] * self.temperatures["6"]) / (
                    self.ammonia_mass["5"] + self.ammonia_mass["2"])

        # E-101 calculations
        Q_zoneI = self.ammonia_mass["1"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.temperatures["8"])
        Q_heatI = self.ammonia_mass["1"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.temperatures["8"]) + ThermalConstants.VAPORIZATION_HEAT * self.ammonia_mass["5"]
        self.temperatures["geo_out"] = temperature + 5
        self.temperatures["geo_out"] = max(self.temperatures["geo_out"], 27)
        geo_mass_flow = Q_heatI / (ThermalConstants.HEAT_CAPACITY * (self.temperatures["geo_in"] - self.temperatures["geo_out"]))
        self.temperatures["I"] = self.temperatures["geo_out"] + Q_zoneI / (geo_mass_flow * ThermalConstants.HEAT_CAPACITY)
        Q_zoneII = Q_heatI - Q_zoneI

        lmtd_1 = [
            log_mean_temperature((self.temperatures["geo_out"] - self.temperatures["8"]), (self.temperatures["I"] - temperature)),
            log_mean_temperature((self.temperatures["I"] - temperature), (self.temperatures["geo_in"] - temperature))
        ]
        area_1 = np.sum([exchanger_area(q, lmtd) for q, lmtd in zip([Q_zoneI, Q_zoneII], lmtd_1)])

        # E-102 calculations
        Q_heat_II = self.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.temperatures["4"]) + ThermalConstants.VAPORIZATION_HEAT * self.ammonia_mass["5"]
        T_out_cold = 20
        cold_mass_flow = Q_heat_II / (ThermalConstants.HEAT_CAPACITY * (T_out_cold - self.temperatures["cw_in"]))
        Q2_zoneII = self.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.temperatures["4"])
        Q2_zoneI = Q_heat_II - Q2_zoneII

        self.temperatures["2I"] = self.temperatures["cw_in"] + Q2_zoneI / (cold_mass_flow * ThermalConstants.HEAT_CAPACITY)

        lmtd_2 = [
            log_mean_temperature((self.temperatures["4"] - self.temperatures["cw_in"]), (temperature - self.temperatures["2I"])),
            log_mean_temperature((temperature - self.temperatures["2I"]), (temperature - self.temperatures["cw_out"]))
        ]
        area_2 = np.sum([exchanger_area(q, lmtd) for q, lmtd in zip([Q2_zoneII, Q2_zoneI], lmtd_2)])

        # E-103 calculations
        chilled_mass_flow = self.energy_demand["Ice Museum"] / ThermalConstants.HEAT_CAPACITY * (self.temperatures["10"] - self.temperatures["9"])

        lmtd_3 = log_mean_temperature((self.temperatures["10"] - self.temperatures["6"]), (self.temperatures["9"] - self.temperatures["5"]))
        area_3 = exchanger_area(self.energy_demand["Ice Museum"], lmtd_3)

        # Cost calculations
        PChx = heat_exchanger_cost(area_1 + area_2 + area_3)
        PCpump = 8000 * (self.ammonia_mass["1"] * (pressure_in_1 - ThermalConstants.PRESSURE_IN_5)) ** 0.6
        PCabsorber = ThermalConstants.ABSORBER_NUMBER * 3000
        anualize_factor = anual_factor(self.rate, self.nper)
        PCtotal = PChx + PCpump + PCabsorber

        UCpump = 0.30 * self.ammonia_mass["1"] * (pressure_in_1 - ThermalConstants.PRESSURE_IN_5) * 8760
        PAC = anualize_factor * PCtotal + UCpump
        EAOC = PAC

        return EAOC, geo_mass_flow, cold_mass_flow, chilled_mass_flow


class OptimizationAnalyzer:
    """Class for analyzing and visualizing optimal operating conditions."""
    
    def __init__(self, system: AbsorberChillingSystem):
        """Initialize with a chilling system."""
        self.system = system
        
    def generate_plot(self, start_temperature: int, end_temperature: int, steps: int) -> tuple[BytesIO, tuple[float]]:
        """Generate optimization plot.
        
        Args:
            start_temperature (int): Starting temperature for optimization.
            end_temperature (int): Ending temperature for optimization.
            steps (int): Number of steps in temperature range.
            
        Returns:
            tuple: (BytesIO buffer containing plot image, optimal values)
        """
        temperature_range = np.linspace(start_temperature, end_temperature, steps)
        eaoc_values = [(temperature, *self.system.calculate_eaoc(temperature)) for temperature in temperature_range]
        optimal = min(eaoc_values, key=lambda x: x[1])
        
        # Styling
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        
        plt.figure(figsize=(4, 4))
        plt.plot(temperature_range, [item[1] for item in eaoc_values], color="#2B47B9", ls='-.', label='EAOC vs Temperature')
        plt.scatter(optimal[0], optimal[1], color='g', label=f'Optimal Point: {optimal[0]:.2f} °C')
        plt.title('Temperature [°C] vs EAOC [$]')
        plt.xlabel('Temperature [°C]')
        plt.ylabel('EAOC [$/year]')
        plt.grid(True, alpha=0.6)
        plt.legend()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return buf, optimal


class AbsorberChillingUI:
    """UI class for the absorber chilling system application."""
    
    def __init__(self, page: ft.Page):
        """Initialize the UI with a Flet page."""
        self.page = page
        self.setup_page()
        self.system = AbsorberChillingSystem()
        self.analyzer = OptimizationAnalyzer(self.system)
        
        # UI components
        self.temperature_start = ft.TextField(label="Start Temperature (°C)", value="30", width=150)
        self.temperature_end = ft.TextField(label="End Temperature (°C)", value="60", width=150)
        self.steps = ft.TextField(label="Steps", value="10000", width=100)
        self.cold_demand = ft.TextField(label="Cooling Demand (kW)", value="500", width=150)
        self.img = ft.Image()
        self.results = ft.Column()
        self.generate_btn = ft.ElevatedButton("Generate Optimization Graph", on_click=self.on_generate)
        
        # New components for temperature profiles
        self.temp_profiles_btn = ft.ElevatedButton("Show Temperature Profiles", on_click=self.on_show_temp_profiles)
        self.exchanger_selector = ft.Dropdown(
            label="Select Heat Exchanger",
            options=[
                ft.dropdown.Option("E-101"),
                ft.dropdown.Option("E-102"),
                ft.dropdown.Option("E-103"),
                ft.dropdown.Option("All")
            ],
            value="All",
            width=150
        )
        self.profile_img = ft.Image()
        
        self.build_ui()
        
    def setup_page(self):
        """Set up the page properties."""
        self.page.title = "Geothermal Absorber Chilling Cost Analysis"
        self.page.window.width = 900
        self.page.window.height = 800
        self.page.window.bgcolor = "system"
        
    def build_ui(self):
        """Build and add UI components to the page."""
        self.page.add(
            ft.Row([self.temperature_start, self.temperature_end, self.steps], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.cold_demand], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.generate_btn]),
            ft.Row([self.img, self.results], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(),
            ft.Row([self.exchanger_selector, self.temp_profiles_btn], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.profile_img], alignment=ft.MainAxisAlignment.CENTER)
        )
        
    async def on_generate(self, e):
        """Handle generate button click event."""
        try:
            start_temperature = float(self.temperature_start.value)
            end_temperature = float(self.temperature_end.value)
            step_count = int(self.steps.value)
            
            # Update cooling demand if changed
            cooling_demand = float(self.cold_demand.value)
            if self.system.energy_demand["Ice Museum"] != cooling_demand:
                self.system.energy_demand = {"Ice Museum": cooling_demand}

            loop = asyncio.get_running_loop()
            buf, optimal = await loop.run_in_executor(None, self.analyzer.generate_plot, 
                                                     start_temperature, end_temperature, step_count)
            self.img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            self.results.controls = [
                ft.Text(f"Optimal Temperature: {optimal[0]:.2f} °C"),
                ft.Text(f"Optimal EAOC: ${optimal[1]:.2f}"),
                ft.Text(f"Geothermal Mass Flow Rate: {optimal[2]:.2f} kg/s"),
                ft.Text(f"Cold Water Mass Flow Rate: {optimal[3]:.2f} kg/s"),
                ft.Text(f"Chilled Mass Flow Rate: {optimal[4]:.2f} kg/s")
            ]

            # Run the calculations at the optimal temperature to update system state
            self.system.calculate_eaoc(optimal[0])
            self.page.update()

        except Exception as ex:
            snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            self.page.overlay.append(snack_bar)
            snack_bar.open = True
            self.page.update()
    
    async def on_show_temp_profiles(self, e):
        """Handle show temperature profiles button click event."""
        try:
            # Make sure we have a temperature to work with
            temperature = float(self.temperature_start.value)
            
            # Perform calculations to ensure all temperature values are up to date
            self.system.calculate_eaoc(temperature)
            
            loop = asyncio.get_running_loop()
            selected = self.exchanger_selector.value
            
            # Generate the temperature profile graph
            buf = await loop.run_in_executor(None, self.generate_temperature_profile, selected, temperature)
            self.profile_img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            self.page.update()
            
        except Exception as ex:
            snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            self.page.overlay.append(snack_bar)
            snack_bar.open = True
            self.page.update()
    
    def generate_temperature_profile(self, exchanger, temperature):
        """Generate temperature profile graphs for the selected heat exchanger(s).
        
        Args:
            exchanger (str): The exchanger to visualize ("E-101", "E-102", "E-103", or "All")
            temperature (float): The operating temperature
            
        Returns:
            BytesIO: Buffer containing the plot image
        """
        # Set up figure size based on how many graphs we're showing
        if exchanger == "All":
            fig, axs = plt.subplots(3, 1, figsize=(6, 10))
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            axs = [ax]
        
        # Configure plot styling
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        
        exchangers_to_plot = ["E-101", "E-102", "E-103"] if exchanger == "All" else [exchanger]
        
        for i, ex in enumerate(exchangers_to_plot):
            ax = axs[i if exchanger == "All" else 0]
            
            if ex == "E-101":
                # Extract E-101 data
                hot_temps = [self.system.temperatures["geo_in"], self.system.temperatures["I"], self.system.temperatures["geo_out"]]
                cold_temps = [self.system.temperatures["8"], temperature, temperature]
                
                # Calculate positions based on heat transfer
                Q_zoneI = self.system.ammonia_mass["1"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.system.temperatures["8"])
                Q_heatI = self.system.ammonia_mass["1"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.system.temperatures["8"]) + ThermalConstants.VAPORIZATION_HEAT * self.system.ammonia_mass["5"]
                Q_zoneII = Q_heatI - Q_zoneI
                
                # Positions based on heat duty
                positions = [0, Q_zoneI, Q_zoneI + Q_zoneII]
                
                # Plot temperature profiles
                ax.plot(positions, hot_temps, 'r-', label='Hot Stream (Geothermal)')
                ax.plot(positions, cold_temps, 'b-', label='Cold Stream (Ammonia)')
                
                # Add labels and annotations
                ax.text(Q_zoneI/2, (hot_temps[1] + hot_temps[2])/2, "Zone I", ha='center')
                ax.text((Q_zoneI + Q_zoneII)/2, (hot_temps[0] + hot_temps[1])/2, "Zone II", ha='center')
                ax.text(Q_zoneI/3, cold_temps[1]-5, "m1", ha='center')
                ax.text(Q_zoneI + Q_zoneII/2, cold_temps[2]-5, "m3", ha='center')
                
            elif ex == "E-102":
                # Extract E-102 data
                hot_temps = [temperature, temperature, self.system.temperatures["4"]]
                cold_temps = [self.system.temperatures["cw_out"], self.system.temperatures["2I"], self.system.temperatures["cw_in"]]
                
                # Calculate positions based on heat transfer
                Q_heat_II = self.system.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.system.temperatures["4"]) + ThermalConstants.VAPORIZATION_HEAT * self.system.ammonia_mass["5"]
                Q2_zoneII = self.system.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - self.system.temperatures["4"])
                Q2_zoneI = Q_heat_II - Q2_zoneII
                
                # Positions based on heat duty
                positions = [0, Q2_zoneI, Q2_zoneI + Q2_zoneII]
                
                # Plot temperature profiles
                ax.plot(positions, hot_temps, 'r-', label='Hot Stream (Ammonia)')
                ax.plot(positions, cold_temps, 'b-', label='Cold Stream (Cooling Water)')
                
                # Add labels and annotations
                ax.text(Q2_zoneI/2, (hot_temps[0] + hot_temps[1])/2 + 2, "Zone I", ha='center')
                ax.text(Q2_zoneI + Q2_zoneII/2, (hot_temps[1] + hot_temps[2])/2 + 2, "Zone II", ha='center')
                
            elif ex == "E-103":
                # Extract E-103 data
                hot_temps = [self.system.temperatures["10"], self.system.temperatures["9"]]
                cold_temps = [self.system.temperatures["6"], self.system.temperatures["5"]]
                
                # Calculate positions based on heat transfer
                Q = self.system.energy_demand["Ice Museum"]
                
                # Positions based on heat duty
                positions = [0, Q]
                
                # Plot temperature profiles
                ax.plot(positions, hot_temps, 'r-', label='Hot Stream (Chilled Water)')
                ax.plot(positions, cold_temps, 'b-', label='Cold Stream (Refrigerant)')
            
            # Set up axis labels and title
            ax.set_xlabel('Heat Transfer [kW]')
            ax.set_ylabel('Temperature [°C]')
            ax.set_title(f'Temperature Profile for {ex}')
            ax.grid(True, alpha=0.6)
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        
        return buf


def main(page: ft.Page):
    """Main application entry point."""
    AbsorberChillingUI(page)


if __name__ == "__main__":
    ft.app(target=main)