from numpy import exp

def heat_exchanger_cost(area: float) -> float:
    """Calculate the cost of a heat exchanger.

    Args:
        area (float): Exchange area in m^2.

    Returns:
        float: Cost of heat exchanger in USD.
    """
    
    return 12000  * area ** 0.57


def turbine_cost(power: float) -> float:
    """Calculate the cost of a turbine.

    Args:
        power (float): Power in kW.

    Returns:
        float: Cost of turbine in USD.
    """
    
    return 5000 * power ** 0.69


def pump_cost(turbine_cost: float) -> float:
    """Calculate the cost of a pump.

    Args:
        turbine_cost (float): Turbine cost in USD.

    Returns:
        float: Pump cost in USD.
    """
    
    return 0.1 * turbine_cost

def well_cost(depth: float) -> float:
    """Calculate the cost of a well.

    Args:
        depth (float): Depth in meters.

    Returns:
        float: Well cost in USD.
    """
    
    return 2 * (250_000 + 696_000 * exp(0.0008 * depth))

def pipe_cost(diameter: float) -> float:
    """Calculate the purchase cost for a pipe line in diameter function.

    Args:
        diameter (float): Pipe diameter [inch]

    Returns:
        float: Pipe cost [$/ft of pipe]
    """
    return 50 * diameter + 5 * diameter ** 1.75


def reactor_cost(volume: float) -> float:
    """Calculate the reactor cost in volume function."""
    return 17000 * volume ** 0.85

