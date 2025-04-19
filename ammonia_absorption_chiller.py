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

        self.temperatures["2I"] = self.temperatures["cw_in"] + Q2_zoneII / (cold_mass_flow * ThermalConstants.HEAT_CAPACITY)

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
        """Generate optimization plot."""
        temperature_range = np.linspace(start_temperature, end_temperature, steps)
        eaoc_values = [(temperature, *self.system.calculate_eaoc(temperature)) for temperature in temperature_range]
        optimal = min(eaoc_values, key=lambda x: x[1])
        
        # Styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
        plt.figure(figsize=(6, 4), dpi=1000)
        plt.plot(temperature_range, [item[1] for item in eaoc_values], color="#2B47B9", lw=2, label='EAOC vs Temperature')
        plt.scatter(optimal[0], optimal[1], color='#32CD32', s=80, zorder=5, label=f'Optimal: {optimal[0]:.2f} °C')
        plt.title('Temperature Optimization Analysis', fontweight='bold')
        plt.xlabel('Temperature [°C]')
        plt.ylabel('EAOC [$/year]')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(frameon=True, framealpha=0.9)
        plt.tight_layout(pad=2.0)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf, optimal


    def generate_temperature_profile(self, exchanger, temperature):
        """Generate temperature profile graphs for the selected heat exchanger(s)."""
        system = self.system

        # Apply consistent plot styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })

        # Set up figure size based on how many graphs we're showing
        if exchanger == "All":
            fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=1000)
            plt.subplots_adjust(hspace=0.4)
        else:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
            axs = [ax]

        exchangers_to_plot = ["E-101", "E-102", "E-103"] if exchanger == "All" else [exchanger]

        for i, ex in enumerate(exchangers_to_plot):
            ax = axs[i if exchanger == "All" else 0]

            if ex == "E-101":
                hot_temps = [system.temperatures["geo_out"], system.temperatures["I"], system.temperatures["geo_in"]]
                cold_temps = [system.temperatures["8"], temperature, temperature]

                Q_zoneI = system.ammonia_mass["1"] * ThermalConstants.HEAT_CAPACITY * (temperature - system.temperatures["8"])
                Q_heatI = Q_zoneI + ThermalConstants.VAPORIZATION_HEAT * system.ammonia_mass["5"]
                Q_zoneII = Q_heatI - Q_zoneI
                positions = [0, Q_zoneI, Q_zoneI + Q_zoneII]

                ax.plot(positions, hot_temps, color='#FF5733', lw=2.5, label='Hot Stream (Geothermal)')
                ax.plot(positions, cold_temps, color='#3498DB', lw=2.5, label='Cold Stream (Ammonia)')

                ax.fill_between(
                    [positions[0], positions[1]],
                    [cold_temps[0], cold_temps[1]],
                    [hot_temps[0], hot_temps[1]],
                    color='#BEB1D9', alpha=0.15
                )
                ax.fill_between(
                    [positions[1], positions[2]],
                    [cold_temps[1], cold_temps[2]],
                    [hot_temps[1], hot_temps[2]],
                    color='#BEB1D9', alpha=0.25
                )

                ax.text(Q_zoneI / 2, (hot_temps[1] + hot_temps[2]) / 3.5, "Zone I",
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                ax.text((Q_zoneI + Q_zoneII) / 2, (hot_temps[0] + hot_temps[1]) / 2, "Zone II",
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            elif ex == "E-102":
                hot_temps = [system.temperatures["4"], temperature, temperature]
                cold_temps = [system.temperatures["cw_in"], system.temperatures["2I"], system.temperatures["cw_out"]]

                Q_heat_II = system.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - system.temperatures["4"]) + ThermalConstants.VAPORIZATION_HEAT * system.ammonia_mass["5"]
                Q2_zoneII = system.ammonia_mass["5"] * ThermalConstants.HEAT_CAPACITY * (temperature - system.temperatures["4"])
                Q2_zoneI = Q_heat_II - Q2_zoneII
                positions = [0, Q2_zoneII, Q2_zoneI + Q2_zoneII]

                ax.plot(positions, hot_temps, color='#FF5733', lw=2.5, label='Hot Stream (Ammonia)')
                ax.plot(positions, cold_temps, color='#3498DB', lw=2.5, label='Cold Stream (Cooling Water)')

                ax.fill_between(
                    [positions[0], positions[1]],
                    [cold_temps[0], cold_temps[1]],
                    [hot_temps[0], hot_temps[1]],
                    color='#BEB1D9', alpha=0.15
                )
                ax.fill_between(
                    [positions[1], positions[2]],
                    [cold_temps[1], cold_temps[2]],
                    [hot_temps[1], hot_temps[2]],
                    color='#BEB1D9', alpha=0.25
                )

                ax.text(Q2_zoneII / 2, (hot_temps[1] + hot_temps[2]) / 5, "Zone I",
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                ax.text((Q2_zoneI + Q2_zoneII) / 2, (hot_temps[0] + hot_temps[1]) / 2, "Zone II",
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                
            elif ex == "E-103":
                hot_temps = [system.temperatures["10"], system.temperatures["9"]]
                cold_temps = [system.temperatures["6"], system.temperatures["5"]]
                Q = system.energy_demand["Ice Museum"]
                positions = [0, Q]

                ax.plot(positions, hot_temps, color='#FF5733', lw=2.5, label='Hot Stream (Chilled Water)')
                ax.plot(positions, cold_temps, color='#3498DB', lw=2.5, label='Cold Stream (Refrigerant)')
                
                ax.fill_between(
                    [positions[0], positions[1]],
                    [cold_temps[0], cold_temps[1]],
                    [hot_temps[0], hot_temps[1]],
                    color='#BEB1D9', alpha=0.15
                )
                ax.text(Q / 2.5, (hot_temps[0] + hot_temps[1]) / 2.5, "Zone",
                        ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # Set axis labels and grid
            ax.set_xlabel('Heat Transfer [kW]', fontweight='bold')
            ax.set_ylabel('Temperature [°C]', fontweight='bold')
            ax.set_title(f'Temperature Profile: {ex}', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', frameon=True, framealpha=0.9)

        plt.tight_layout()

        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        return buf


class AbsorberChillingUI:
    """UI class for the absorber chilling system application."""
    
    def __init__(self, page: ft.Page):
        """Initialize the UI with a Flet page."""
        self.page = page
        self.setup_page()
        self.system = AbsorberChillingSystem()
        self.analyzer = OptimizationAnalyzer(self.system)
        
        # Create all UI components
        self.create_ui_components()
        self.build_ui()
        
    def setup_page(self):
        """Set up the page properties."""
        self.page.title = "Geothermal Absorber Chilling Analysis"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 20
        self.page.window.width = 1000
        self.page.window.height = 850
        self.page.window.bgcolor = "#f5f5f5"
        self.page.fonts = {
            "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
        }
        self.page.expand = True
        
    def create_ui_components(self):
        """Create all UI components."""
        # Header
        self.header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.AC_UNIT_ROUNDED, size=28, color="#2B47B9"),
                ft.Text("Geothermal Absorber Chilling System", 
                       size=24, weight=ft.FontWeight.BOLD, color="#2B47B9"),
            ], alignment=ft.MainAxisAlignment.CENTER),
            padding=5,
            margin=ft.margin.only(bottom=5),
            border_radius=10,
            bgcolor=ft.colors.WHITE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.colors.BLACK12,
                offset=ft.Offset(0, 2),
            )
        )
        
        # Input section components
        self.temperature_start = ft.TextField(
            label="Start Temperature (°C)",
            value="30",
            width=200,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.THERMOSTAT_OUTLINED,
            helper_text="Lower bound"
        )
        
        self.temperature_end = ft.TextField(
            label="End Temperature (°C)",
            value="60",
            width=200,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.THERMOSTAT,
            helper_text="Upper bound"
        )
        
        self.steps = ft.TextField(
            label="Steps",
            value="10000",
            width=100,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.TUNE,
            helper_text="Precision"
        )
        
        self.cold_demand = ft.TextField(
            label="Cooling Demand (kW)",
            value="500",
            width=200,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.POWER,
            helper_text="System capacity"
        )
        
        self.generate_btn = ft.ElevatedButton(
            "Run Optimization",
            icon=ft.icons.PLAY_ARROW_ROUNDED,
            on_click=self.on_generate,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                color=ft.colors.WHITE,
                bgcolor="#2B47B9",
                padding=5
            )
        )
        
        # Results components
        self.img = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            border_radius=8,
            expand=True
        )
        
        self.results = ft.Container(
            content=ft.Column(
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.START
            ),
            padding=15,
            border_radius=8,
            bgcolor=ft.colors.WHITE,
            width=320,
            visible=False,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.colors.BLACK12,
                offset=ft.Offset(0, 2),
            )
        )
        
        # Temperature profiles components
        self.exchanger_selector = ft.Dropdown(
            label="Select Heat Exchanger",
            hint_text="Choose exchanger to visualize",
            options=[
                ft.dropdown.Option("E-101"),
                ft.dropdown.Option("E-102"),
                ft.dropdown.Option("E-103"),
                ft.dropdown.Option("All")
            ],
            value="All",
            width=200,
            height=40,
            text_size=14,
            prefix_icon=ft.icons.DEVICE_HUB,
            content_padding=ft.padding.symmetric(vertical=14, horizontal=12),
        )
        
        self.temp_profiles_btn = ft.ElevatedButton(
            "Show Temperature Profiles",
            icon=ft.icons.INSIGHTS_ROUNDED,
            on_click=self.on_show_temp_profiles,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                color=ft.colors.WHITE,
                bgcolor="#2B47B9",
                padding=5
            )
        )
        
        self.profile_img = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            border_radius=8
        )
        
        # Main tabs
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Cost Optimization",
                    icon=ft.icons.QUERY_STATS,
                    content=ft.Container(
                        content=self.build_optimization_tab(),
                        padding=10
                    )
                ),
                ft.Tab(
                    text="Temperature Profiles",
                    icon=ft.icons.INSIGHTS,
                    content=ft.Container(
                        content=self.build_profiles_tab(),
                        padding=10
                    )
                ),
            ],
            expand=True
        )
        
    def build_optimization_tab(self):
        """Build the optimization tab content."""
        input_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("System Parameters", weight=ft.FontWeight.BOLD, size=10),
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Row([
                        self.cold_demand
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Text("Optimization Settings", weight=ft.FontWeight.BOLD, size=10),
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Row([
                        self.temperature_start,
                        self.temperature_end,
                        self.steps
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row([
                        self.generate_btn
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=15)
                ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=5
            ),
            elevation=2
        )
        
        results_view = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Container(content=self.img, expand=True),
                    self.results
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=20, vertical_alignment=ft.CrossAxisAlignment.START),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            margin=ft.margin.only(top=20),
            expand=True
        )
        
        return ft.Column([
            input_panel,
            results_view
        ], spacing=10, expand=True)
    
    def build_profiles_tab(self):
        """Build the temperature profiles tab content."""
        controls_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Temperature Profile Visualization", weight=ft.FontWeight.BOLD, size=14),
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Row([
                        self.exchanger_selector,
                        self.temp_profiles_btn
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=15)
                ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=5
            ),
            elevation=2
        )
        
        profile_view = ft.Container(
            content=self.profile_img,
            margin=ft.margin.only(top=20),
            alignment=ft.alignment.center,
            expand=True
        )
        
        return ft.Column([
            controls_panel,
            profile_view
        ], spacing=10, expand=True)

    def build_ui(self):
        """Build and add UI components to the page."""
        self.page.add(
        ft.Column([
            self.header,
            self.tabs
        ])
        )
        
    async def on_generate(self, e):
        """Handle generate button click event."""
        try:
            # Show loading state
            self.generate_btn.text = "Calculating..."
            self.generate_btn.disabled = True
            self.page.update()
            
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

            self.results.visible = True
            self.results.content = ft.Column([
                ft.Text("Optimization Results", size=18, weight=ft.FontWeight.BOLD, color="#2B47B9"),
                ft.Divider(height=0.5, color="#E0E0E0"),
                ft.Row([
                    ft.Icon(ft.icons.THERMOSTAT, color="#2B47B9"),
                    ft.Text(f"Optimal Temperature: ", weight=ft.FontWeight.BOLD),
                    ft.Text(f"{optimal[0]:.2f} °C")
                ]),
                ft.Row([
                    ft.Icon(ft.icons.ATTACH_MONEY, color="#2B47B9"),
                    ft.Text(f"Optimal EAOC: ", weight=ft.FontWeight.BOLD),
                    ft.Text(f"${optimal[1]:.2f}")
                ]),
                ft.Divider(height=0.5, color="#E0E0E0"),
                ft.Text("Mass Flow Rates:", weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.icons.WATER_DROP, color="#e65100"),
                            ft.Text(f"Geothermal: ", weight=ft.FontWeight.BOLD),
                            ft.Text(f"{optimal[2]:.2f} kg/s")
                        ]),
                        ft.Row([
                            ft.Icon(ft.icons.WATER_DROP, color="#0277bd"),
                            ft.Text(f"Cold Water: ", weight=ft.FontWeight.BOLD),
                            ft.Text(f"{optimal[3]:.2f} kg/s")
                        ]),
                        ft.Row([
                            ft.Icon(ft.icons.WATER_DROP, color="#00838f"),
                            ft.Text(f"Chilled: ", weight=ft.FontWeight.BOLD),
                            ft.Text(f"{optimal[4]:.2f} kg/s")
                        ])
                    ], spacing=10),
                    padding=10,
                    bgcolor="#f5f5f5",
                    border_radius=6
                )
            ], spacing=10)

            # Run the calculations at the optimal temperature to update system state
            self.system.calculate_eaoc(optimal[0])
            
            # Reset button state
            self.generate_btn.text = "Run Optimization"
            self.generate_btn.disabled = False
            self.page.update()

        except Exception as ex:
            # Reset button state
            self.generate_btn.text = "Run Optimization"
            self.generate_btn.disabled = False
            
            snack_bar = ft.SnackBar(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.ERROR_OUTLINE, color=ft.colors.WHITE),
                        ft.Text(f"Error: {ex}", color=ft.colors.WHITE)
                    ]),
                    padding=10
                ),
                bgcolor="#d32f2f"
            )
            if snack_bar not in self.page.overlay:
                self.page.overlay.append(snack_bar)
            snack_bar.open = True
            self.page.update()

    
    async def on_show_temp_profiles(self, e):
        """Handle show temperature profiles button click event."""
        try:
            # Show loading state
            self.temp_profiles_btn.text = "Generating..."
            self.temp_profiles_btn.disabled = True
            self.page.update()
            
            # Make sure we have a temperature to work with
            temperature = float(self.temperature_start.value)
            
            # Perform calculations to ensure all temperature values are up to date
            self.system.calculate_eaoc(temperature)
            
            loop = asyncio.get_running_loop()
            selected = self.exchanger_selector.value
            
            # Generate the temperature profile graph
            buf = await loop.run_in_executor(None, self.analyzer.generate_temperature_profile, selected, temperature)
            self.profile_img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            # Reset button state
            self.temp_profiles_btn.text = "Show Temperature Profiles"
            self.temp_profiles_btn.disabled = False
            self.page.update()
            
        except Exception as ex:
            # Reset button state
            self.temp_profiles_btn.text = "Show Temperature Profiles"
            self.temp_profiles_btn.disabled = False
            
            snack_bar = ft.SnackBar(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.ERROR_OUTLINE, color=ft.colors.WHITE),
                        ft.Text(f"Error: {ex}", color=ft.colors.WHITE)
                    ]),
                    padding=10
                ),
                bgcolor="#d32f2f"
            )
            self.page.snack_bar = snack_bar
            snack_bar.open = True
            self.page.update()


def main(page: ft.Page):
    """Main application entry point."""
    AbsorberChillingUI(page)


if __name__ == "__main__":
    ft.app(target=main)