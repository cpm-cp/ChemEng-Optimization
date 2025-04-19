import numpy as np

def log_mean_temperature(DT1: float, DT2: float) -> float:
    """Calculate the logarithm mean temperature difference in counter current.

    Args:
        DT1 (float): (Warm out - Cold in)
        DT2 (float): (Warm in - Cold out)

    Returns:
        float: log mean temperature difference
    """
    if DT1 == DT2:  # To avoid division by zero
        return DT1
    return (DT1 - DT2) / np.log(DT1 / DT2)

def exchanger_area(heat: float, lmtd: float, U: float = 1, F: float = 0.8) -> float:
    """Calculate the exchanger area.    

    Args:
        heat (float): Heat flow rate in kW.
        lmtd (float): Log mean tempeature difference.
        U (float, optional): Global heat transfer coeffcient . Defaults to 1.
        F (float, optional): Correction factor. Defaults to 0.8.

    Returns:
        float: Exchanger area in meters.
    """
    return heat / (F * U * lmtd)

def friction_factor(reynold: float, diameter:float, roughness: float = 0.045e-3) -> float:
    """Calculate the friction factor by the Swamee-Jain equation.

    Args:
        reynold (float): dimensionaless reynold number.
        diameter (float): pipe diameter [m]
        roughness (float, optional): pipe roughness. Defaults to 0.045e-3.

    Returns:
        float: Dimensionaless friction factor.
    """
    return (1 / (-4 * np.log10((roughness / (3.7 * diameter)) + (6.81 / reynold)**0.9)))**2