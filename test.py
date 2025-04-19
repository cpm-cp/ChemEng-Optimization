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
        
        # Temperature profiles for exchangers (simulated data)
        self.generate_exchanger_profiles()
        
    def generate_exchanger_profiles(self):
        """Generate simulated temperature profiles for exchangers."""
        self.exchanger_profiles = {}
        
        # E-101: Geothermal heat exchanger
        self.exchanger_profiles["E-101"] = self.generate_profile(
            base_hot=75, base_cold=25, 
            label_hot="Geothermal Fluid", 
            label_cold="Ammonia-Water Solution",
            description="Geothermal Heat Exchanger"
        )
        
        # E-102: Cooling water heat exchanger
        self.exchanger_profiles["E-102"] = self.generate_profile(
            base_hot=40, base_cold=4.5, 
            label_hot="Ammonia-Water Solution",
            label_cold="Cooling Water",
            description="Cooling Water Heat Exchanger"
        )
        
        # E-103: Evaporator
        self.exchanger_profiles["E-103"] = self.generate_profile(
            base_hot=-5, base_cold=-25, 
            label_hot="Chilled Water", 
            label_cold="Ammonia",
            description="Evaporator"
        )
    
    def generate_profile(self, base_hot, base_cold, label_hot, label_cold, description, points=20):
        """Generate temperature profile data for an exchanger."""
        # Position along exchanger (0-100%)
        positions = np.linspace(0, 100, points)
        
        # Temperature profiles with some randomness
        hot_profile = base_hot - (base_hot - base_cold + 10) * (positions/100)**1.2 + np.random.normal(0, 0.5, points)
        cold_profile = base_cold + (base_hot - base_cold - 10) * (positions/100)**0.8 + np.random.normal(0, 0.3, points)
        
        return {
            "positions": positions,
            "hot_profile": hot_profile,
            "cold_profile": cold_profile,
            "label_hot": label_hot,
            "label_cold": label_cold,
            "description": description
        }
        
    @staticmethod
    def pressure(temperature: float) -> float:
        """Calculate pressure as temperature function."""
        return 0.26 * np.exp(0.025 * temperature)
    
    @staticmethod
    def water_composition(temperature: float) -> float:
        """Water composition as temperature function."""
        return (1.89 * temperature + 1.41) / 10_000
    
    @staticmethod
    def liquid_mass_ratio(water_composition: float) -> float:
        """m_2/m_1 liquid mass ratio."""
        return 0.905 - 8.39 * (water_composition * water_composition) - 0.708 * water_composition
    
    def calculate_eaoc(self, temperature: float) -> list[float]:
        """Calculate EAOC for a given temperature."""
        # Initial calculations
        pressure_in_1 = self.pressure(temperature)
        water_comp = self.water_composition(temperature)
        mass_ratio = self.liquid_mass_ratio(water_comp)
        
        # Mass calculations
        self.ammonia_mass["5"] = self.energy_demand["Ice Museum"] / ThermalConstants.VAPORIZATION_HEAT
        self.ammonia_mass["1"] = self.ammonia_mass["5"] / (1 - mass_ratio)
        self.ammonia_mass["2"] = self.ammonia_mass["1"] - self.ammonia_mass["5"]
        self.temperatures["8"] = (self.ammonia_mass["2"] * temperature + self.ammonia_mass["5"] * self.temperatures["6"]) / (self.ammonia_mass["5"] + self.ammonia_mass["2"])

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
    
    def generate_temperature_profile_plot(self, exchanger_id):
        """Generate temperature profile plot for a specific exchanger."""
        if exchanger_id not in self.exchanger_profiles:
            return None
        
        profile = self.exchanger_profiles[exchanger_id]
        
        # Styling
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        
        plt.figure(figsize=(10, 6))
        plt.plot(profile["positions"], profile["hot_profile"], 'r-', linewidth=2, label=profile["label_hot"])
        plt.plot(profile["positions"], profile["cold_profile"], 'b-', linewidth=2, label=profile["label_cold"])
        
        plt.title(f'Temperature Profile - {exchanger_id}: {profile["description"]}')
        plt.xlabel('Position in Heat Exchanger (%)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.6)
        plt.legend()
        
        # Fill between curves to show temperature difference
        plt.fill_between(profile["positions"], profile["hot_profile"], profile["cold_profile"], 
                         color='lightgray', alpha=0.5)
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close()
        
        return buf


class OptimizationAnalyzer:
    """Class for analyzing and visualizing optimal operating conditions."""
    
    def __init__(self, system: AbsorberChillingSystem):
        """Initialize with a chilling system."""
        self.system = system
        
    def generate_plot(self, start_temperature: int, end_temperature: int, steps: int) -> tuple[BytesIO, tuple[float]]:
        """Generate optimization plot."""
        temperature_range = np.linspace(start_temperature, end_temperature, steps)
        eaoc_values = [(temperature, *self.system.calculate_eaoc(temperature)) for temperature in temperature_range]
        optimal = min(eaoc_values, key=lambda x: x[1])
        
        # Styling
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 12
        })
        
        plt.figure(figsize=(8, 6))
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


class TemperatureProfileWindow:
    """Class for displaying temperature profile in a separate window."""
    
    def __init__(self, system: AbsorberChillingSystem):
        """Initialize with a chilling system."""
        self.system = system
    
    async def show_profile(self, page: ft.Page):
        """Display temperature profile in a separate window."""
        # Get exchanger ID from URL parameters
        params = page.route_url_params
        exchanger_id = params.get("exchanger", "E-101")
        
        # Setup page properties
        page.title = f"Temperature Profile - {exchanger_id}"
        page.window_width = 900
        page.window_height = 700
        page.padding = 20
        
        # Generate temperature profile plot
        loop = asyncio.get_running_loop()
        buf = await loop.run_in_executor(None, self.system.generate_temperature_profile_plot, exchanger_id)
        
        if not buf:
            page.add(ft.Text(f"Error: Could not generate temperature profile for {exchanger_id}"))
            return
        
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        
        # Get profile data for displaying additional information
        profile = self.system.exchanger_profiles[exchanger_id]
        
        # Create temperature profile image
        img = ft.Image(
            src_base64=img_str,
            width=800,
            height=500,
            fit=ft.ImageFit.CONTAIN
        )
        
        # Create data table for temperature values
        data_table = ft.DataTable(
            border=ft.border.all(1, ft.colors.GREY_400),
            border_radius=10,
            horizontal_lines=ft.border.BorderSide(1, ft.colors.GREY_300),
            columns=[
                ft.DataColumn(ft.Text("Position (%)")),
                ft.DataColumn(ft.Text(f"{profile['label_hot']} (°C)")),
                ft.DataColumn(ft.Text(f"{profile['label_cold']} (°C)")),
                ft.DataColumn(ft.Text("ΔT (°C)"))
            ],
            rows=[
                ft.DataRow(cells=[
                    ft.DataCell(ft.Text(f"{pos:.0f}")),
                    ft.DataCell(ft.Text(f"{hot:.1f}")),
                    ft.DataCell(ft.Text(f"{cold:.1f}")),
                    ft.DataCell(ft.Text(f"{hot - cold:.1f}"))
                ])
                for pos, hot, cold in zip(
                    profile["positions"][::5],  # Sample every 5th point
                    profile["hot_profile"][::5],
                    profile["cold_profile"][::5]
                )
            ]
        )
        
        # Add close button
        close_button = ft.ElevatedButton(
            "Close",
            icon=ft.icons.CLOSE,
            on_click=lambda _: page.window_close()
        )
        
        # Add components to page
        page.add(
            ft.Row([
                ft.Text(f"Temperature Profile - {exchanger_id}", size=24, weight=ft.FontWeight.BOLD),
                ft.Container(width=20),
                close_button
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(height=20),
            img,
            ft.Container(height=20),
            ft.Text("Temperature Data Sample Points:", size=16, weight=ft.FontWeight.BOLD),
            ft.Container(height=10),
            data_table
        )


class AbsorberChillingUI:
    """UI class for the absorber chilling system application."""
    
    def __init__(self, page: ft.Page):
        """Initialize the UI with a Flet page."""
        self.page = page
        self.setup_page()
        self.system = AbsorberChillingSystem()
        self.analyzer = OptimizationAnalyzer(self.system)
        self.profile_window = TemperatureProfileWindow(self.system)
        
        # UI components
        self.temperature_start = ft.TextField(label="Start Temperature (°C)", value="30", width=150)
        self.temperature_end = ft.TextField(label="End Temperature (°C)", value="60", width=150)
        self.steps = ft.TextField(label="Steps", value="10000", width=100)
        self.cold_demand = ft.TextField(label="Cooling Demand (kW)", value="500", width=150)
        self.img = ft.Image()
        self.results = ft.Column()
        self.generate_btn = ft.ElevatedButton("Generate Graph", on_click=self.on_generate)
        
        # Add exchanger section
        self.exchanger_section = self.build_exchanger_section()
        
        self.build_ui()
        
    def setup_page(self):
        """Set up the page properties."""
        self.page.title = "Geothermal Absorber Chilling Cost Analysis"
        self.page.window.width = 1000
        self.page.window.height = 800
        self.page.window.bgcolor = "system"
        
    def build_exchanger_section(self):
        """Build the exchanger selection section."""
        exchanger_cards = ft.Row(
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            controls=[
                self.create_exchanger_card(exchanger_id) 
                for exchanger_id in ["E-101", "E-102", "E-103"]
            ]
        )
        
        return ft.Container(
            padding=20,
            margin=ft.margin.only(top=20, bottom=20),
            border=ft.border.all(1, ft.colors.GREY_400),
            border_radius=10,
            content=ft.Column(
                spacing=10,
                controls=[
                    ft.Text("Heat Exchanger Temperature Profiles", size=18, weight=ft.FontWeight.BOLD),
                    ft.Text("Click on an exchanger to view its temperature profile in a separate window."),
                    exchanger_cards
                ]
            )
        )
        
    def create_exchanger_card(self, exchanger_id):
        """Create a card for an exchanger."""
        description = self.system.exchanger_profiles[exchanger_id]["description"]
        
        return ft.Card(
            elevation=3,
            content=ft.Container(
                width=250,
                padding=15,
                content=ft.Column(
                    spacing=15,
                    controls=[
                        ft.Text(exchanger_id, size=20, weight=ft.FontWeight.BOLD),
                        ft.Text(description, size=14),
                        ft.FilledButton(
                            "View Temperature Profile",
                            icon=ft.icons.SHOW_CHART,
                            on_click=lambda e, id=exchanger_id: self.open_profile_window(id)
                        )
                    ]
                )
            )
        )
    
    def open_profile_window(self, exchanger_id):
        """Open a new window with temperature profile."""
        self.page.launch_url(f"/profile?exchanger={exchanger_id}")
        
    def build_ui(self):
        """Build and add UI components to the page."""
        self.page.add(
            ft.Text("Geothermal Absorber Chilling Cost Analysis", size=24, weight=ft.FontWeight.BOLD),
            ft.Container(height=10),
            ft.Text("Optimization Parameters", size=16, weight=ft.FontWeight.BOLD),
            ft.Row([self.temperature_start, self.temperature_end, self.steps], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.cold_demand], alignment=ft.MainAxisAlignment.CENTER),
            ft.Row([self.generate_btn]),
            ft.Container(height=20),
            ft.Row([self.img, self.results], alignment=ft.MainAxisAlignment.CENTER),
            self.exchanger_section
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

            self.page.update()

        except Exception as ex:
            snack_bar = ft.SnackBar(ft.Text(f"Error: {ex}"), bgcolor="red")
            self.page.overlay.append(snack_bar)
            snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    """Main application entry point."""
    # Handle routing between main view and profile view
    def route_change(e):
        page.views.clear()
        
        if page.route == "/" or page.route == "":
            page.views.append(
                ft.View(
                    route="/",
                    controls=[],
                    padding=0
                )
            )
            AbsorberChillingUI(page)
            
        elif page.route.startswith("/profile"):
            page.views.append(
                ft.View(
                    route="/profile",
                    controls=[],
                    padding=0
                )
            )
            system = AbsorberChillingSystem()
            profile_window = TemperatureProfileWindow(system)
            asyncio.create_task(profile_window.show_profile(page))
            
        page.update()
    
    # Configure page routing
    page.on_route_change = route_change
    
    # Initialize the app
    page.go(page.route)


if __name__ == "__main__":
    # Start the app with routing enabled
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, assets_dir="assets")