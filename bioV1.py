import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
use("agg")
import flet as ft
from costs.operational_cost import anual_factor
from costs.equipment_cost import heat_exchanger_cost, reactor_cost
from costs.tools import exchanger_area, log_mean_temperature
from typing import Final
from io import BytesIO
import base64
import asyncio
from dataclasses import dataclass


GAL_PER_HOUR_TO_CUBIC_METERS: Final[float] = 1.0515 * 10**-6
CONVERSION: Final[float] = 0.8
VOLUMENTRIC_FLOW: Final[float] = 5000 * GAL_PER_HOUR_TO_CUBIC_METERS # m^3/s

@dataclass
class Substance:
    density: float
    caloric_capacity: float
    in_temperature: float | None
    out_temperature: float | None
    
    
@dataclass
class SubstanceParameters(Substance):
    COLD = Substance(density=1000, caloric_capacity=4.18, in_temperature=20, out_temperature=None)
    HOT = Substance(density=920, caloric_capacity=2.2, in_temperature=65, out_temperature=30)
    
    
class BioreactorSystem:
    """Class representing the bioreactor system."""
    
    def __init__(self, conversion=None):
        """Initialize the bioreactor system with conversion."""
        self.conversion = conversion or CONVERSION
        self.rate = 0.07
        self.nper = 12
        
    def calculate_eaoc(self, out_temperature: float) -> list[float]:
        """Calculate EAOC for a given cold out temperature."""
        
        absolute_temperature = out_temperature + 273.15 # K
        k_reaction = 9.5 * np.exp(-3300 / absolute_temperature)
        volume = (VOLUMENTRIC_FLOW * self.conversion) / (k_reaction * (1 - self.conversion))
        
        # Heat flow in kW
        mass_flow = VOLUMENTRIC_FLOW * SubstanceParameters.COLD.density
        heat_flow = mass_flow * SubstanceParameters.COLD.caloric_capacity * (out_temperature - SubstanceParameters.COLD.in_temperature)
        lmtd = log_mean_temperature((SubstanceParameters.HOT.out_temperature - SubstanceParameters.COLD.in_temperature), (SubstanceParameters.HOT.in_temperature - out_temperature))
        area = exchanger_area(heat_flow, lmtd, 0.4)
        PChx = heat_exchanger_cost(area)
        
        PCrx = reactor_cost(volume)
        anualize_factor = anual_factor(self.rate, self.nper)
        PCtotal = PChx + PCrx
        UC = 5e-6 * heat_flow * 3600 * 8760
        EAOC = anualize_factor * PCtotal + UC
        
        return EAOC, mass_flow, PChx, PCrx, UC
    
class OptimizationAnalyzer:
    """Class for analyzing and visualizing optimal operating conditions."""
    
    def __init__(self, system: BioreactorSystem):
        self.system = system
        
    def generate_plot(self, start_temperature: int, end_temperature:int, steps: int, 
                      show_eaoc=True, show_pchx=False, show_pcrx=False, show_uc=False) -> tuple[BytesIO, dict]:
        """Generate optimization plot with multiple lines based on visibility settings."""
        
        temperature_range = np.linspace(start_temperature, end_temperature, steps)
        calculated_values = [(temperature, *self.system.calculate_eaoc(temperature)) for temperature in temperature_range]
        
        # Extract individual data series
        eaoc_values = [(temp, eaoc) for temp, eaoc, _, _, _, _ in calculated_values]
        pchx_values = [(temp, pchx) for temp, _, _, pchx, _, _ in calculated_values]
        pcrx_values = [(temp, pcrx) for temp, _, _, _, pcrx, _ in calculated_values]
        uc_values = [(temp, uc) for temp, _, _, _, _, uc in calculated_values]
        
        # Find optimal points for each data series
        optimal_eaoc = min(eaoc_values, key=lambda x: x[1]) if eaoc_values else None
        optimal_pchx = min(pchx_values, key=lambda x: x[1]) if pchx_values else None
        optimal_pcrx = min(pcrx_values, key=lambda x: x[1]) if pcrx_values else None
        optimal_uc = min(uc_values, key=lambda x: x[1]) if uc_values else None
        
        # Store all optimal values
        optimal_values = {
            'eaoc': optimal_eaoc,
            'pchx': optimal_pchx,
            'pcrx': optimal_pcrx,
            'uc': optimal_uc
        }
        
        # Styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })   
        
        plt.figure(figsize=(8, 6), dpi=1000)
        
        # Plot each line based on visibility flags
        if show_eaoc:
            plt.plot(temperature_range, [item[1] for item in calculated_values], 
                    color="#2B47B9", lw=2, label='EAOC vs Temperature')
            plt.scatter(optimal_eaoc[0], optimal_eaoc[1], color='#2B47B9', s=80, zorder=5)
        
        if show_pchx:
            plt.plot(temperature_range, [item[3] for item in calculated_values], 
                    color="#FF5722", lw=2, label='PChx vs Temperature')
            plt.scatter(optimal_pchx[0], optimal_pchx[1], color='#FF5722', s=80, zorder=5)
        
        if show_pcrx:
            plt.plot(temperature_range, [item[4] for item in calculated_values], 
                    color="#4CAF50", lw=2, label='PCrx vs Temperature')
            plt.scatter(optimal_pcrx[0], optimal_pcrx[1], color='#4CAF50', s=80, zorder=5)
        
        if show_uc:
            plt.plot(temperature_range, [item[5] for item in calculated_values], 
                    color="#9C27B0", lw=2, label='UC vs Temperature')
            plt.scatter(optimal_uc[0], optimal_uc[1], color='#9C27B0', s=80, zorder=5)
        
        plt.title('Temperature Optimization Analysis', fontweight='bold')
        plt.xlabel('Out Temperature [Cold] [°C]')
        plt.ylabel('Cost [$]')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(frameon=True, framealpha=0.9)
        plt.tight_layout(pad=2.0)

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Get the raw data for each calculation to return
        temps = list(temperature_range)
        eaoc_data = [item[1] for item in calculated_values]
        pchx_data = [item[3] for item in calculated_values]
        pcrx_data = [item[4] for item in calculated_values]
        uc_data = [item[5] for item in calculated_values]
        
        # Calculate the full results for a single optimal temperature point (using EAOC optimal)
        if optimal_eaoc:
            optimal_temp = optimal_eaoc[0]
            full_result = next((item for item in calculated_values if item[0] == optimal_temp), None)
        else:
            full_result = calculated_values[0] if calculated_values else None
        
        return buf, optimal_values, full_result
    
    
class BioreactorUI:
    """UI class for the bioreactor system application."""
    
    def __init__(self, page: ft.Page):
        """Initialize the UI with a flet Page."""
        self.page = page
        self.setup_page()
        self.system = BioreactorSystem()
        self.analyzer = OptimizationAnalyzer(self.system)
        
        # Visibility flags for different plot lines
        self.show_eaoc = True
        self.show_pchx = False
        self.show_pcrx = False
        self.show_uc = False
        
        # Store optimal values
        self.optimal_values = {}
        self.full_result = None
        
        # Create all UI components
        self.create_ui_components()
        self.build_ui()
        
    def setup_page(self):
        """Set up the page properties."""
        self.page.title = "Bioreactor Analysis System"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.padding = 15
        self.page.window.width = 1200
        self.page.window.height = 850
        self.page.fonts = {
            "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
        }
        self.page.expand = True
        
    def create_ui_components(self):
        """Create all UI components."""
        # Header
        self.header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.icons.API, size=22, color="#2B47B9"),
                ft.Text("Advanced Bioprocess System Analysis", size=20, weight=ft.FontWeight.BOLD, color="#2B47B9"),
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
            label="Start Temperature [°C]",
            value="25",
            width=220,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.THERMOSTAT_AUTO_OUTLINED,
            helper_text="Lower bound"
        )
        
        self.temperature_end = ft.TextField(
            label="End temperature [°C]",
            value="64",
            width=220,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.THERMOSTAT,
            helper_text="Upper bound"
        )
        
        self.steps = ft.TextField(
            label="Steps",
            value="2000",
            width=120,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.TUNE,
            helper_text="Precision"
        )
        
        self.conversion = ft.TextField(
            label="Reactant conversion [%]",
            value="0.8",
            width=200,
            height=50,
            text_size=12,
            border_radius=6,
            prefix_icon=ft.icons.ROTATE_LEFT_SHARP,
            helper_text="Conversion rate"
        )
        
        self.generate_btn = ft.ElevatedButton(
            "Run optimization",
            icon=ft.icons.PLAY_ARROW_ROUNDED,
            on_click=self.on_generate,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                color=ft.colors.WHITE,
                bgcolor="#2B47B9",
                padding=5
            )
        )
        
        # Plot toggle switches
        self.eaoc_switch = ft.Switch(
            label="EAOC vs Temperature",
            value=True,
            active_color="#2B47B9",
            on_change=self.toggle_eaoc
        )
        
        self.pchx_switch = ft.Switch(
            label="PChx vs Temperature",
            value=False,
            active_color="#FF5722",
            on_change=self.toggle_pchx
        )
        
        self.pcrx_switch = ft.Switch(
            label="PCrx vs Temperature",
            value=False,
            active_color="#4CAF50",
            on_change=self.toggle_pcrx
        )
        
        self.uc_switch = ft.Switch(
            label="UC vs Temperature",
            value=False,
            active_color="#9C27B0",
            on_change=self.toggle_uc
        )
        
        # Result components
        self.img = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            border_radius=8,
            expand=True
        )
        
        self.results = ft.Container(
            content=ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.START
            ),
            padding=5,
            border_radius=8,
            bgcolor=ft.colors.WHITE,
            width=270,
            visible=False,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=12,
                color=ft.colors.BLACK12,
                offset=ft.Offset(0, 2),
            )
        )
        
        # Optimal values display
        self.optimal_container = ft.Container(
            content=ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.START
            ),
            padding=10,
            border_radius=8,
            bgcolor=ft.colors.WHITE,
            visible=False,
            expand=True,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=12,
                color=ft.colors.BLACK12,
                offset=ft.Offset(0, 2),
            )
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
                    text="Advanced Analysis",
                    icon=ft.icons.INSIGHTS,
                    content=ft.Container(
                        content=self.build_analysis_tab(),
                        padding=10
                    )
                )
            ],
            expand=True
        )
        
    def build_optimization_tab(self):
        """Build the optimization tab content."""
        input_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("System parameters", weight=ft.FontWeight.BOLD, size=14),
                    ft.Divider(height=0.4, color="#E0E0E0"),
                    ft.Row([
                        self.conversion
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Text("Optimization Settings", weight=ft.FontWeight.BOLD, size=14),
                    ft.Divider(height=0.4, color="#E0E0E0"),
                    ft.Row([
                        self.temperature_start,
                        self.temperature_end,
                        self.steps
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Row([
                        self.generate_btn
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=10)
                ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=5
            ),
            elevation=2
        )
        
        # Plot toggle panel
        plot_toggle_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Plot Options", weight=ft.FontWeight.BOLD, size=14),
                    ft.Divider(height=0.4, color="#E0E0E0"),
                    ft.Row([
                        ft.Column([
                            ft.Container(
                                content=ft.Row([
                                    self.eaoc_switch,
                                    ft.Container(width=18),
                                    self.pchx_switch
                                ]),
                                padding=5
                            ),
                            ft.Container(
                                content=ft.Row([
                                    self.pcrx_switch,
                                    ft.Container(width=18),
                                    self.uc_switch
                                ]),
                                padding=5
                            )
                        ], spacing=0)
                    ], alignment=ft.MainAxisAlignment.CENTER)
                ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=5
            ),
            elevation=2
        )
        
        result_view = ft.Container(
            content=ft.Column([
                plot_toggle_panel,
                ft.Row([
                    ft.Container(content=self.img, expand=True),
                    self.results
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=15, vertical_alignment=ft.CrossAxisAlignment.START),
                self.optimal_container
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            margin=ft.margin.only(top=10),
            expand=True
        )
        
        return ft.Column([
            input_panel,
            result_view
        ], spacing=10, expand=True)
        
    def build_analysis_tab(self):
        """Build the analysis tab content."""
        analysis_panel = ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Analysis Graph", weight=ft.FontWeight.BOLD, size=14),
                    ft.Divider(height=0.4, color="#E0E0E0"),
                    ft.Text("Select plot options from the Cost Optimization tab to see analysis")
                ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=10
            ),
            elevation=2
        )
        return analysis_panel
        
    def build_ui(self):
        """Build and add UI components to the page."""
        self.page.add(
            ft.Column([
                self.header,
                self.tabs
            ])
        )
    
    # FIX: These toggle methods need to call an async method properly
    def toggle_eaoc(self, e):
        """Toggle EAOC plot visibility."""
        self.show_eaoc = e.control.value
        if self.full_result:
            asyncio.create_task(self.update_plot())  # Use create_task instead of direct call
    
    def toggle_pchx(self, e):
        """Toggle PChx plot visibility."""
        self.show_pchx = e.control.value
        if self.full_result:
            asyncio.create_task(self.update_plot())  # Use create_task instead of direct call
    
    def toggle_pcrx(self, e):
        """Toggle PCrx plot visibility."""
        self.show_pcrx = e.control.value
        if self.full_result:
            asyncio.create_task(self.update_plot())  # Use create_task instead of direct call
    
    def toggle_uc(self, e):
        """Toggle UC plot visibility."""
        self.show_uc = e.control.value
        if self.full_result:
            asyncio.create_task(self.update_plot())  # Use create_task instead of direct call
    
    async def update_plot(self):
        """Update the plot based on current toggle settings."""
        try:
            start_temperature = float(self.temperature_start.value)
            end_temperature = float(self.temperature_end.value)
            step_count = int(self.steps.value)
            
            loop = asyncio.get_running_loop()
            buf, optimal_values, full_result = await loop.run_in_executor(
                None, 
                lambda: self.analyzer.generate_plot(
                    start_temperature, 
                    end_temperature, 
                    step_count, 
                    self.show_eaoc, 
                    self.show_pchx, 
                    self.show_pcrx, 
                    self.show_uc
                )
            )
            
            # Update the image
            self.img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            # Store the optimal values and full result
            self.optimal_values = optimal_values
            self.full_result = full_result
            
            # Update the optimal values display
            self.update_optimal_display()
            
            self.page.update()
        except Exception as ex:
            self.show_error(f"Error updating plot: {ex}")
    
    def update_optimal_display(self):
        """Update the optimal values display panel."""
        if not self.optimal_values:
            return
            
        columns = []
        
        # Create columns for each visible plot line
        if self.show_eaoc and 'eaoc' in self.optimal_values and self.optimal_values['eaoc']:
            eaoc_opt = self.optimal_values['eaoc']
            columns.append(
                ft.Column([
                    ft.Container(
                        content=ft.Text("EAOC Optimal", weight=ft.FontWeight.BOLD, color="#2B47B9", size=16),
                        padding=5,
                        bgcolor="#E3F2FD",
                        border_radius=4
                    ),
                    ft.Row([
                        ft.Icon(ft.icons.THERMOSTAT, color="#2B47B9"),
                        ft.Text(f"Temperature: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{eaoc_opt[0]:.2f} °C")
                    ]),
                    ft.Row([
                        ft.Icon(ft.icons.ATTACH_MONEY, color="#2B47B9"),
                        ft.Text(f"Value: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{eaoc_opt[1]:.2f} $/yr")
                    ])
                ], spacing=5)
            )
        
        if self.show_pchx and 'pchx' in self.optimal_values and self.optimal_values['pchx']:
            pchx_opt = self.optimal_values['pchx']
            columns.append(
                ft.Column([
                    ft.Container(
                        content=ft.Text("PChx Optimal", weight=ft.FontWeight.BOLD, color="#FF5722", size=16),
                        padding=5,
                        bgcolor="#FBE9E7",
                        border_radius=4
                    ),
                    ft.Row([
                        ft.Icon(ft.icons.THERMOSTAT, color="#FF5722"),
                        ft.Text(f"Temperature: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{pchx_opt[0]:.2f} °C")
                    ]),
                    ft.Row([
                        ft.Icon(ft.icons.ATTACH_MONEY, color="#FF5722"),
                        ft.Text(f"Value: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{pchx_opt[1]:.2f} $")
                    ])
                ], spacing=5)
            )
            
        if self.show_pcrx and 'pcrx' in self.optimal_values and self.optimal_values['pcrx']:
            pcrx_opt = self.optimal_values['pcrx']
            columns.append(
                ft.Column([
                    ft.Container(
                        content=ft.Text("PCrx Optimal", weight=ft.FontWeight.BOLD, color="#4CAF50", size=16),
                        padding=5,
                        bgcolor="#E8F5E9",
                        border_radius=4
                    ),
                    ft.Row([
                        ft.Icon(ft.icons.THERMOSTAT, color="#4CAF50"),
                        ft.Text(f"Temperature: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{pcrx_opt[0]:.2f} °C")
                    ]),
                    ft.Row([
                        ft.Icon(ft.icons.ATTACH_MONEY, color="#4CAF50"),
                        ft.Text(f"Value: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{pcrx_opt[1]:.2f} $")
                    ])
                ], spacing=5)
            )
            
        if self.show_uc and 'uc' in self.optimal_values and self.optimal_values['uc']:
            uc_opt = self.optimal_values['uc']
            columns.append(
                ft.Column([
                    ft.Container(
                        content=ft.Text("UC Optimal", weight=ft.FontWeight.BOLD, color="#9C27B0", size=16),
                        padding=5,
                        bgcolor="#F3E5F5",
                        border_radius=4
                    ),
                    ft.Row([
                        ft.Icon(ft.icons.THERMOSTAT, color="#9C27B0"),
                        ft.Text(f"Temperature: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{uc_opt[0]:.2f} °C")
                    ]),
                    ft.Row([
                        ft.Icon(ft.icons.ATTACH_MONEY, color="#9C27B0"),
                        ft.Text(f"Value: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{uc_opt[1]:.2f} $/yr")
                    ])
                ], spacing=5)
            )
        
        # Only show the optimal container if we have data to display
        if columns:
            self.optimal_container.visible = True
            self.optimal_container.content = ft.Column([
                ft.Text("Optimal Values", size=18, weight=ft.FontWeight.BOLD),
                ft.Divider(height=0.5, color="#E0E0E0"),
                ft.Row(columns, alignment=ft.MainAxisAlignment.SPACE_AROUND)
            ], spacing=10)
        else:
            self.optimal_container.visible = False
    
    async def on_generate(self, e):
        """Handle generate button click event."""
        try:
            self.generate_btn.text = "Calculating..."
            self.generate_btn.disabled = True
            self.page.update()
            
            start_temperature = float(self.temperature_start.value)
            end_temperature = float(self.temperature_end.value)
            step_count = int(self.steps.value)
            
            # Update conversion if changed
            conversion = float(self.conversion.value)
            if self.system.conversion != conversion:
                self.system.conversion = conversion
                
            loop = asyncio.get_running_loop()
            buf, optimal_values, full_result = await loop.run_in_executor(
                None, 
                lambda: self.analyzer.generate_plot(
                    start_temperature, 
                    end_temperature, 
                    step_count, 
                    self.show_eaoc, 
                    self.show_pchx, 
                    self.show_pcrx, 
                    self.show_uc
                )
            )
            
            self.img.src_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            
            # Store the results
            self.optimal_values = optimal_values
            self.full_result = full_result
            
            # Show the results panel with EAOC overview
            self.results.visible = True
            eaoc_optimal = optimal_values.get('eaoc')
            if eaoc_optimal and full_result:
                self.results.content = ft.Column([
                    ft.Text("Optimization Results", size=16, weight=ft.FontWeight.BOLD, color="#2B47B9"),
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Row([
                        ft.Icon(ft.icons.THERMOSTAT, color="#2B47B9"),
                        ft.Text(f"Optimal Temperature: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{eaoc_optimal[0]:.2f} °C")
                    ]),
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Row([
                        ft.Icon(ft.icons.WATER_DROP, color="#e65100"),
                        ft.Text(f"Mass Flow: ", weight=ft.FontWeight.BOLD),
                        ft.Text(f"{full_result[2]:.2f} kg/s")
                    ]),                
                    ft.Divider(height=0.5, color="#E0E0E0"),
                    ft.Text("Complete Cost Analysis:", weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.icons.ATTACH_MONEY, color="#2B47B9"),
                                ft.Text(f"EAOC: ", weight=ft.FontWeight.BOLD),
                                ft.Text(f"{full_result[1]:.2f} $/yr")
                            ]),
                            ft.Row([
                                ft.Icon(ft.icons.ATTACH_MONEY, color="#FF5722"),
                                ft.Text(f"PChx: ", weight=ft.FontWeight.BOLD),
                                ft.Text(f"{full_result[3]:.2f} $")
                            ]),
                            ft.Row([
                                ft.Icon(ft.icons.ATTACH_MONEY, color="#4CAF50"),
                                ft.Text(f"PCrx: ", weight=ft.FontWeight.BOLD),
                                ft.Text(f"{full_result[4]:.2f} $")
                            ]),
                            ft.Row([
                                ft.Icon(ft.icons.ATTACH_MONEY, color="#9C27B0"),
                                ft.Text(f"UC: ", weight=ft.FontWeight.BOLD),
                                ft.Text(f"{full_result[5]:.2f} $/yr")
                            ])
                        ], spacing=15),
                        padding=15,
                        bgcolor="#f5f5f5",
                        border_radius=6
                    )
                ], spacing=15)
            
            # Update the optimal values display
            self.update_optimal_display()
            
            # Reset button state
            self.generate_btn.text = "Run Optimization"
            self.generate_btn.disabled = False
            self.page.update()
            
        except Exception as ex:
            self.show_error(f"Error: {ex}")
            # Reset button state
            self.generate_btn.text = "Run Optimization"
            
    
    def show_error(self, message: str):
        """Show error message in a snack bar."""
        # Reset button state if needed
        self.generate_btn.text = "Run Optimization"
        self.generate_btn.disabled = False
        
        snack_bar = ft.SnackBar(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.ERROR_OUTLINE, color=ft.colors.WHITE),
                    ft.Text(message, color=ft.colors.WHITE)
                ]),
                padding=10
            ),
            bgcolor="#d32f2f"
        )
        if snack_bar not in self.page.overlay:
            self.page.overlay.append(snack_bar)
        snack_bar.open = True
        self.page.update()
        
    def on_generate_analysis(self, e):
        """Handle generate analysis button click event."""
        # This is a placeholder for future functionality
        if self.full_result:
            snack_bar = ft.SnackBar(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.icons.INFO_OUTLINE, color=ft.colors.WHITE),
                        ft.Text("Analysis is shown in the Cost Optimization tab", color=ft.colors.WHITE)
                    ]),
                    padding=10
                ),
                bgcolor="#2196f3"
            )
            if snack_bar not in self.page.overlay:
                self.page.overlay.append(snack_bar)
            snack_bar.open = True
            self.page.update()
        else:
            self.show_error("Please run optimization first")


def main(page: ft.Page):
    """Main entry point."""
    BioreactorUI(page)
    

if __name__ == "__main__":
    ft.app(target=main)