def operational_cost(demand:float, product: float) -> float:
    return (demand - product) * 0.08 * 8000


def anual_factor(rate:float, nper:int) -> float:
    return (rate * (1 + rate)**nper) / ((1 + rate)**nper - 1) 