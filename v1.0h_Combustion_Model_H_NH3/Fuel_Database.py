import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable

O2_mass_fraction_in_air = 0.233  # mass fraction of O2 in air

@dataclass
class FuelSpec:
    name: str
    LHV_MJ_per_kg: float           # Lower Heating Value
    O2_req_per_kg: float           # kg of O2 required per kg of fuel
    AFR_stoich: float              # mass of air per mass fuel
    MW_g_per_mol: float            # molecular weight
    rho_gas_kg_per_m3: float       # density at 1 atm, 20°C
    flam_limits_volpct: tuple      # (LFL, UFL)
    S_L_300K_m_per_s: float        # laminar flame speed at 300 K, 1 atm
    T_autoign_K: float             # autoignition temperature
    gamma_u_func: Optional[Callable[[float], float]] = None  # γ(T) function

    def gamma_u(self, T_K: float = 300.0) -> float:
        if self.gamma_u_func:
            return self.gamma_u_func(T_K)
        return 1.35  # fallback default
class FuelDB:
    def __init__(self):
        self.db: Dict[str, FuelSpec] = {}

    def register(self, name: str, spec: FuelSpec) -> None:
        self.db[name] = spec

    def get(self, name: str) -> FuelSpec:
        if name not in self.db:
            raise KeyError(f"Fuel '{name}' not found. Available: {list(self.db.keys())}")
        return self.db[name]

    def list(self) -> List[str]:
        return sorted(self.db.keys())

# ---------------------------------------------------------------------
# --- FUNCTIONS
# ---------------------------------------------------------------------
def gamma_H2(T: float) -> float:
    g = 1.40 - 1.2e-4 * (T - 300.0)
    return max(1.25, min(1.40, g))

def gamma_NH3(T: float) -> float:
    g = 1.39 - 1.0e-4 * (T - 300.0)
    return max(1.22, min(1.39, g))
# ---------------------------------------------------------------------
# --- DATABASE ENTRIES
# ---------------------------------------------------------------------
Fuels = FuelDB()
Fuels.register(
    "H2",
    FuelSpec(
        name="Hydrogen",
        LHV_MJ_per_kg=120.0,
        O2_req_per_kg=8.000,
        AFR_stoich=8.000 / O2_mass_fraction_in_air,
        MW_g_per_mol=2.016,
        rho_gas_kg_per_m3=0.084,
        flam_limits_volpct=(4.0, 75.0),
        S_L_300K_m_per_s=2.5,
        T_autoign_K=858.0,
        gamma_u_func=gamma_H2
    ),
)
Fuels.register(
    "NH3",
    FuelSpec(
        name="Ammonia",
        LHV_MJ_per_kg=18.6,
        O2_req_per_kg=1.412,
        AFR_stoich=1.412 / O2_mass_fraction_in_air,
        MW_g_per_mol=17.031,
        rho_gas_kg_per_m3=0.73,
        flam_limits_volpct=(15.0, 28.0),
        S_L_300K_m_per_s=0.07,
        T_autoign_K=924.0,
        gamma_u_func=gamma_NH3
    ),
)
Fuels.register(
    "Gasoline",
    FuelSpec(
        name="Gasoline",
        LHV_MJ_per_kg=43.0,
        O2_req_per_kg=14.7 * O2_mass_fraction_in_air,
        AFR_stoich=14.7,
        MW_g_per_mol=100.0,
        rho_gas_kg_per_m3=0.74,
        flam_limits_volpct=(1.4, 7.6),
        S_L_300K_m_per_s=0.37,
        T_autoign_K=803.0,
        gamma_u_func=lambda T: max(1.20, min(1.38, 1.38 - 1.1e-4 * (T - 300.0))),
    ),
)
Fuels.register(
    "CH4",
    FuelSpec(
        name="Methane",
        LHV_MJ_per_kg=50.0,
        O2_req_per_kg=17.2 * O2_mass_fraction_in_air,
        AFR_stoich=17.2,
        MW_g_per_mol=16.04,
        rho_gas_kg_per_m3=0.67,
        flam_limits_volpct=(5.0, 15.0),
        S_L_300K_m_per_s=0.38,
        T_autoign_K=813.0,
        gamma_u_func=lambda T: max(1.22, min(1.40, 1.39 - 1.0e-4 * (T - 300.0))),
    ),
)
# ---------------------------------------------------------------------
# --- H2–NH3 BLENDING
# ---------------------------------------------------------------------
def blend_H2_NH3(w_H2: float) -> Dict[str, float]:
    """
    Blend H2 and NH3 by mass fraction (0–1) and compute effective properties.
    Returns a dict of key parameters.
    """
    w_H2 = np.clip(w_H2, 0.0, 1.0)

    H2 = Fuels.get("H2")
    NH3 = Fuels.get("NH3")

    O2_req = w_H2 * H2.O2_req_per_kg + (1 - w_H2) * NH3.O2_req_per_kg
    AFR = O2_req / O2_mass_fraction_in_air
    LHV = w_H2 * H2.LHV_MJ_per_kg + (1 - w_H2) * NH3.LHV_MJ_per_kg
    SL = (1 - w_H2) * NH3.S_L_300K_m_per_s + w_H2 * H2.S_L_300K_m_per_s
    gamma_300 = (1 - w_H2) * NH3.gamma_u(300) + w_H2 * H2.gamma_u(300)

    return {
        "w_H2": w_H2,
        "LHV_MJ_per_kg": LHV,
        "AFR_stoich": AFR,
        "O2_req_per_kg": O2_req,
        "S_L_300K_m_per_s": SL,
        "gamma_300": gamma_300,
    }
def mass_fraction_from_energy_fraction(x_e_H2: float) -> float:
    """
    Convert hydrogen energy fraction (fraction of fuel energy from H2)
    into mass fraction of H2.
    """
    x_e_H2 = np.clip(x_e_H2, 0.0, 1.0)
    H2 = Fuels.get("H2")
    NH3 = Fuels.get("NH3")
    inv_e_H2 = 1.0 / H2.LHV_MJ_per_kg
    inv_e_NH3 = 1.0 / NH3.LHV_MJ_per_kg
    w_H2 = (x_e_H2 * inv_e_H2) / (x_e_H2 * inv_e_H2 + (1 - x_e_H2) * inv_e_NH3)
    return w_H2
