import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
from typing import Final, List, Tuple, Dict, Optional
from io import BytesIO
import base64
import flet as ft
import asyncio

class ProcessConstants:
    """Constants used in the pipe optimization process."""
    VISCOSITY: Final[float] = 0.01  # kg/m*s
    VOLUMETRIC_FLOW: float = 50_000  # barrels/day - not Final to allow modification
    PIPE_LENGTH: float = 100  # miles - not Final to allow modification
    DENSITY: Final[int] = 930  # kg/m^3
    PUMP_EFFICIENCY: Final[float] = 0.85
    
    # Conversion factors
    MILES_TO_METERS: Final[float] = 1609.34
    METERS_TO_INCH: Final[float] = 39.3701
    METERS_TO_FEET: Final[float] = 3.28084
    BARRELS_TO_CUBIC_METERS: Final[float] = 0.00000184
    
    # Economic parameters
    INTEREST_RATE: Final[float] = 0.09
    PROJECT_LIFE: Final[int] = 15  # years
    ELECTRICITY_COST: Final[float] = 0.05  # $/kWh
    HOURS_PER_YEAR: Final[int] = 8760

class Utilities:
    """Utility functions for calculations"""
    
    @staticmethod
    def friction_factor(reynolds: float, diameter: float) -> float:
        """Calculate the friction factor using the Colebrook equation approximation"""
        roughness = 0.0000457  # m, typical for commercial steel pipes
        relative_roughness = roughness / diameter
        
        # Use the Chen approximation of the Colebrook equation
        term1 = -4 * np.log10(relative_roughness/3.7 + (6.81/reynolds)**0.9)
        return (1/term1)**2
    
    @staticmethod
    def annual_factor(rate: float, years: int) -> float:
        """Calculate the capital recovery factor"""
        return rate * (1 + rate)**years / ((1 + rate)**years - 1)
    
    @staticmethod
    def pipe_cost(diameter_inch: float) -> float:
        """Calculate pipe cost per foot based on diameter in inches"""
        return 50 * diameter_inch + 5 * diameter_inch ** 1.75
    
    @staticmethod
    def pump_cost(power_kw: float) -> float:
        """Calculate pump cost based on power in kW"""
        return 8000 * (power_kw)**0.6

class PipelineSystem:
    """Class representing the pipeline system with economic analysis capabilities."""
    
    def __init__(self):
        """Initialize the pipeline system with standard commercial diameters."""
        # Standard commercial diameters in meters
        self.std_diameters = [0.05251, 0.06271, 0.07793, 0.09012, 0.10226, 
                             0.12819, 0.15405, 0.20272, 0.25451, 0.30323, 
                             0.33336, 0.381, 0.42865, 0.47788, 0.5747]
        
        # Custom diameter range
        self.min_diameter = 0.05
        self.max_diameter = 0.60
        self.diameter_step = 0.01
        
        # Default to standard diameters
        self.use_standard_diameters = True
        self.custom_diameter_count = 30
    
    def get_diameters(self) -> List[float]:
        """Get the diameter list based on current settings"""
        if self.use_standard_diameters:
            return self.std_diameters
        else:
            return np.linspace(self.min_diameter, self.max_diameter, self.custom_diameter_count).tolist()
    
    def calculate_eaoc(self, diameter: float) -> Tuple[float, float, float]:
        """Calculate the Equivalent Annual Operating Cost for a given diameter"""
        # Convert to SI units
        volumetric_flow = ProcessConstants.VOLUMETRIC_FLOW * ProcessConstants.BARRELS_TO_CUBIC_METERS
        pipe_length = ProcessConstants.PIPE_LENGTH * ProcessConstants.MILES_TO_METERS
        pipe_length_ft = pipe_length * ProcessConstants.METERS_TO_FEET
        diameter_inch = diameter * ProcessConstants.METERS_TO_INCH
        
        # Calculate Reynolds number
        reynolds_number = 4 * volumetric_flow * ProcessConstants.DENSITY / (np.pi * diameter * ProcessConstants.VISCOSITY)
        
        # Calculate friction factor
        f_factor = Utilities.friction_factor(reynolds_number, diameter)
        
        # Calculate pipe cost
        pipe_capital_cost = Utilities.pipe_cost(diameter_inch) * pipe_length_ft
        
        # Calculate pump power and cost
        pump_power = 32 * f_factor * ProcessConstants.DENSITY * volumetric_flow**3 * pipe_length / (
            ProcessConstants.PUMP_EFFICIENCY * np.pi**2 * diameter**5)
        pump_power_kw = pump_power / 1000
        pump_capital_cost = Utilities.pump_cost(pump_power_kw)
        
        # Calculate annual costs
        annual_factor = Utilities.annual_factor(ProcessConstants.INTEREST_RATE, ProcessConstants.PROJECT_LIFE)
        annual_capital_cost = annual_factor * (pipe_capital_cost + pump_capital_cost)
        annual_operating_cost = ProcessConstants.ELECTRICITY_COST * pump_power_kw * ProcessConstants.HOURS_PER_YEAR
        
        # Total equivalent annual operating cost
        eaoc = annual_capital_cost + annual_operating_cost
        
        return eaoc, diameter_inch, pump_power_kw

class OptimizationAnalyzer:
    """Class to analyze and visualize optimal operating conditions."""
    
    def __init__(self, system: PipelineSystem):
        """Initialize with a pipeline system."""
        self.system = system
        self.results = []
    
    def analyze(self) -> Tuple[BytesIO, Dict]:
        """Analyze the system and generate optimization results"""
        diameters = self.system.get_diameters()
        self.results.clear()
        
        for diameter in diameters:
            eaoc, diam_inch, power_kw = self.system.calculate_eaoc(diameter)
            self.results.append({
                "diameter_m": diameter,
                "diameter_inch": diam_inch,
                "eaoc": eaoc,
                "power_kw": power_kw
            })
        
        # Find optimal point
        optimal = min(self.results, key=lambda x: x["eaoc"])
        
        # Generate plot
        plot_buffer = self._generate_plot(diameters, [r["eaoc"] for r in self.results], optimal)
        
        return plot_buffer, optimal
    
    def _generate_plot(self, diameters: List[float], eaoc_values: List[float], optimal: Dict) -> BytesIO:
        """Generate a plot of EAOC vs diameter"""
        # Styling
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        # Plot the data
        ax.plot(diameters, eaoc_values, color='#3366CC', linewidth=2)
        ax.scatter(optimal["diameter_m"], optimal["eaoc"], color='#FF5733', s=100, 
                  label=f'Optimal: {optimal["diameter_m"]:.3f} m')
        
        # Set labels and title
        ax.set_title('Pipeline Diameter Optimization', fontweight='bold', pad=15)
        ax.set_xlabel('Diameter [m]', fontweight='bold')
        ax.set_ylabel('Annual Cost [$/year]', fontweight='bold')
        
        # Formatting
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        return buf

class PipeOptimizationApp:
    """Main application class for the UI"""
    
    def __init__(self):
        self.pipeline = PipelineSystem()
        self.analyzer = OptimizationAnalyzer(self.pipeline)
        
    def build_ui(self, page: ft.Page):
        """Build the UI for the application"""
        # Set page properties
        page.title = "Pipeline Diameter Optimization"
        page.window.width = 1000
        page.window.height = 800
        page.theme_mode = ft.ThemeMode.LIGHT
        page.padding = 20
        page.window.bgcolor = "#f5f5f5"
        page.fonts = {
            "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
        }
        
        # Application title
        title = ft.Text("Pipeline Diameter Optimization", size=28, weight=ft.FontWeight.BOLD)
        subtitle = ft.Text("Minimize costs by finding the optimal pipe diameter", size=16, italic=True)
        
        # Text fields for input parameters
        self.flow_rate_input = ft.TextField(
            label="Flow Rate (bbl/day)",
            value=str(ProcessConstants.VOLUMETRIC_FLOW),
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
            suffix_text="bbl/day",
            text_align=ft.TextAlign.RIGHT,
            helper_text="Range: 10,000 - 100,000"
        )
        
        self.pipe_length_input = ft.TextField(
            label="Pipe Length (miles)",
            value=str(ProcessConstants.PIPE_LENGTH),
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
            suffix_text="miles",
            text_align=ft.TextAlign.RIGHT,
            helper_text="Range: 10 - 500"
        )
        
        # Input validation text
        self.validation_text = ft.Text(
            "",
            color=ft.colors.RED_500,
            size=12
        )
        
        # Diameter selection
        diameter_toggle = ft.Switch(
            label="Use standard commercial diameters",
            value=self.pipeline.use_standard_diameters,
            on_change=self.toggle_diameter_type
        )
        
        # Custom diameter options (visible when standard diameters are not used)
        self.custom_diameter_count_input = ft.TextField(
            label="Number of diameter points",
            value=str(self.pipeline.custom_diameter_count),
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
            text_align=ft.TextAlign.RIGHT,
            helper_text="Range: 5 - 100"
        )
        
        self.min_diameter_input = ft.TextField(
            label="Minimum Diameter (m)",
            value=str(self.pipeline.min_diameter),
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
            suffix_text="m",
            text_align=ft.TextAlign.RIGHT
        )
        
        self.max_diameter_input = ft.TextField(
            label="Maximum Diameter (m)",
            value=str(self.pipeline.max_diameter),
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
            suffix_text="m",
            text_align=ft.TextAlign.RIGHT
        )
        
        # Custom diameter container (initially hidden)
        self.custom_diameter_container = ft.Container(
            content=ft.Column([
                ft.Text("Custom Diameter Range", weight=ft.FontWeight.BOLD),
                self.min_diameter_input,
                self.max_diameter_input,
                self.custom_diameter_count_input
            ], spacing=10),
            padding=10,
            border_radius=8,
            bgcolor=ft.colors.BLUE_100,
            visible=not self.pipeline.use_standard_diameters
        )
        
        # Results display
        self.image_view = ft.Image(
            width=800,
            height=500,
            fit=ft.ImageFit.CONTAIN,
        )
        
        self.results_text = ft.Column(spacing=10)
        
        # Calculate button
        calculate_btn = ft.ElevatedButton(
            "Calculate Optimal Diameter",
            icon=ft.icons.CALCULATE,
            on_click=self.calculate_optimal,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=10),
                color=ft.colors.WHITE,
                bgcolor=ft.colors.BLUE_600,
                padding=15,
            )
        )
        
        # Layout
        parameters_section = ft.Container(
            content=ft.Column([
                ft.Text("System Parameters", weight=ft.FontWeight.BOLD, size=18),
                self.flow_rate_input,
                self.pipe_length_input,
                self.validation_text,
                ft.Divider(),
                diameter_toggle,
                self.custom_diameter_container,
                ft.Divider(),
                calculate_btn,
            ], spacing=15),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.BLUE_50,
            width=400,
        )
        
        results_section = ft.Container(
            content=ft.Column([
                ft.Text("Optimization Results", weight=ft.FontWeight.BOLD, size=18),
                self.results_text,
            ], spacing=10),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.BLUE_50,
            width=400,
        )
        
        plot_section = ft.Container(
            content=self.image_view,
            padding=10,
            border_radius=10,
            bgcolor=ft.colors.WHITE,
        )
        
        # Main layout
        page.add(
            ft.Column([
                ft.Row([title], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([subtitle], alignment=ft.MainAxisAlignment.CENTER),
                ft.Divider(),
                ft.Row([
                    parameters_section,
                    ft.VerticalDivider(),
                    ft.Column([
                        plot_section,
                        results_section,
                    ], spacing=20),
                ], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
            ], spacing=10)
        )
    
    def toggle_diameter_type(self, e):
        """Toggle between standard and custom diameters"""
        self.pipeline.use_standard_diameters = e.control.value
        self.custom_diameter_container.visible = not e.control.value
        e.page.update()
    
    def validate_inputs(self) -> bool:
        """Validate all input fields and return True if all are valid"""
        try:
            # Validate flow rate
            flow_rate = float(self.flow_rate_input.value)
            if not (10000 <= flow_rate <= 100000):
                self.validation_text.value = "Flow rate must be between 10,000 and 100,000 bbl/day"
                return False
            
            # Validate pipe length
            pipe_length = float(self.pipe_length_input.value)
            if not (10 <= pipe_length <= 500):
                self.validation_text.value = "Pipe length must be between 10 and 500 miles"
                return False
            
            # If custom diameters are used, validate those inputs
            if not self.pipeline.use_standard_diameters:
                min_diam = float(self.min_diameter_input.value)
                max_diam = float(self.max_diameter_input.value)
                count = int(self.custom_diameter_count_input.value)
                
                if min_diam >= max_diam:
                    self.validation_text.value = "Minimum diameter must be less than maximum diameter"
                    return False
                
                if not (0.01 <= min_diam <= 1.0):
                    self.validation_text.value = "Minimum diameter must be between 0.01 and 1.0 meters"
                    return False
                
                if not (0.01 <= max_diam <= 1.0):
                    self.validation_text.value = "Maximum diameter must be between 0.01 and 1.0 meters"
                    return False
                
                if not (5 <= count <= 100):
                    self.validation_text.value = "Number of diameter points must be between 5 and 100"
                    return False
                
                # Update pipeline parameters
                self.pipeline.min_diameter = min_diam
                self.pipeline.max_diameter = max_diam
                self.pipeline.custom_diameter_count = count
            
            # All validations passed, update process constants
            ProcessConstants.VOLUMETRIC_FLOW = flow_rate
            ProcessConstants.PIPE_LENGTH = pipe_length
            
            # Clear validation messages
            self.validation_text.value = ""
            return True
            
        except ValueError:
            self.validation_text.value = "Please enter valid numbers for all fields"
            return False
    
    async def calculate_optimal(self, e):
        """Calculate the optimal diameter and update the UI"""
        # Validate inputs first
        if not self.validate_inputs():
            e.page.update()
            return
            
        e.control.disabled = True
        e.page.update()
        
        # Show progress indicator
        progress = ft.ProgressRing()
        e.page.add(progress)
        e.page.update()
        
        # Run analysis in a separate thread to avoid UI freezing
        def run_analysis():
            return self.analyzer.analyze()
        
        # Use asyncio to run the analysis without blocking the UI
        try:
            image_buffer, optimal = await asyncio.to_thread(run_analysis)
            
            # Update image
            self.image_view.src_base64 = base64.b64encode(image_buffer.read()).decode("utf-8")
            image_buffer.close()
            
            # Format currency
            formatted_eaoc = "${:,.2f}".format(optimal["eaoc"])
            
            # Update results text
            self.results_text.controls = [
                ft.Text(f"Optimal diameter: {optimal['diameter_m']:.4f} m ({optimal['diameter_inch']:.2f} inches)", 
                      size=14, weight=ft.FontWeight.BOLD),
                ft.Text(f"Annual cost: {formatted_eaoc}/year", size=14),
                ft.Text(f"Pump power required: {optimal['power_kw']:.2f} kW", size=14),
                ft.Text(f"Flow rate: {ProcessConstants.VOLUMETRIC_FLOW:,.2f} bbl/day", size=14),
                ft.Text(f"Pipe length: {ProcessConstants.PIPE_LENGTH:,.2f} miles", size=14),
            ]
            
        except Exception as ex:
            # Handle any errors during analysis
            self.validation_text.value = f"Error during calculation: {str(ex)}"
        
        finally:
            # Remove progress indicator and re-enable button
            e.page.remove(progress)
            e.control.disabled = False
            e.page.update()

def main(page: ft.Page):
    app = PipeOptimizationApp()
    app.build_ui(page)

if __name__ == "__main__":
    ft.app(target=main)