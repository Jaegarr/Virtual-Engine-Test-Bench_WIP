import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable, Tuple
O2_mass_fraction_in_air = 0.233  # mass fraction of O2 in dry air
@dataclass
class FuelSpec:
    name: str
    LHV_MJ_per_kg: float                                     # Lower Heating Value [MJ/kg]
    O2_req_per_kg: float                                     # kg O2 required per kg fuel
    AFR_stoich: float                                        # mass air per mass fuel at stoich
    MW_g_per_mol: float                                      # molecular weight [g/mol]
    rho_gas_kg_per_m3: float                                 # gas density @ 1 atm, 20°C [kg/m3] (rarely used)
    flam_limits_volpct: Tuple[float,float]                   # (LFL, UFL) in vol%
    S_L_300K_m_per_s: float                                  # laminar flame speed @300K, 1 atm [m/s]
    T_autoign_K: float                                       # autoignition temperature [K]
    gamma_u_func: Optional[Callable[[float], float]] = None  # γ(T) for unburned mix
    phi_peak: Optional[float] = None                         # φ at which S_L peaks 
    phi_width: Optional[float] = None                        # width of φ peak shape
    alpha_T: Optional[float] = None                          # S_L ~ (T/300)^alpha_T
    beta_p: Optional[float] = None                           # S_L ~ (p/1atm)^beta_p
    phi_minmax: Optional[Tuple[float,float]] = None          # clip φ to [min,max]

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
# γ(T) FUNCTIONS
def gamma_H2(T: float) -> float:
    g = 1.40 - 1.2e-4 * (T - 300.0)
    return max(1.25, min(1.40, g))
def gamma_NH3(T: float) -> float:
    g = 1.39 - 1.0e-4 * (T - 300.0)
    return max(1.22, min(1.39, g))
Fuels = FuelDB()
Fuels.register(
    "Gasoline",
    FuelSpec(
        name="Gasoline",
        LHV_MJ_per_kg=43.0,
        O2_req_per_kg=14.7 * O2_mass_fraction_in_air,  # ≈3.42 kg O2 / kg fuel
        AFR_stoich=14.7,
        MW_g_per_mol=100.0,
        rho_gas_kg_per_m3=0.74,                        # liquid-density proxy kept for legacy usage
        flam_limits_volpct=(1.4, 7.6),
        S_L_300K_m_per_s=0.37,
        T_autoign_K=803.0,
        gamma_u_func=lambda T: max(1.20, min(1.38, 1.38 - 1.1e-4 * (T - 300.0))),
        # optional FS overrides (tune later if desired)
        phi_peak=1.10, phi_width=0.35, alpha_T=2.0, beta_p=-0.25, phi_minmax=(0.5, 1.8),
    ),
)
Fuels.register(
    "Ethanol",
    FuelSpec(
        name="Ethanol",
        LHV_MJ_per_kg=26.8,
        O2_req_per_kg=9.0 * O2_mass_fraction_in_air,   # ≈2.10 kg O2 / kg fuel
        AFR_stoich=9.0,
        MW_g_per_mol=46.07,
        rho_gas_kg_per_m3=1.59,                        # placeholder
        flam_limits_volpct=(3.3, 19.0),
        S_L_300K_m_per_s=0.45,
        T_autoign_K=690.0,
        gamma_u_func=lambda T: max(1.20, min(1.38, 1.37 - 1.0e-4*(T-300))),
        phi_peak=1.10, phi_width=0.35, alpha_T=2.1, beta_p=-0.25, phi_minmax=(0.6, 2.0),
    ),
)
Fuels.register(
    "Methanol",
    FuelSpec(
        name="Methanol",
        LHV_MJ_per_kg=19.9,
        O2_req_per_kg=6.4 * O2_mass_fraction_in_air,   # ≈1.49 kg O2 / kg fuel
        AFR_stoich=6.4,
        MW_g_per_mol=32.04,
        rho_gas_kg_per_m3=1.37,                        # placeholder
        flam_limits_volpct=(6.7, 36.0),
        S_L_300K_m_per_s=0.48,
        T_autoign_K=760.0,
        gamma_u_func=lambda T: max(1.20, min(1.38, 1.37 - 1.0e-4*(T-300))),
        phi_peak=1.10, phi_width=0.35, alpha_T=2.1, beta_p=-0.25, phi_minmax=(0.6, 2.2),
    ),
)
Fuels.register(
    "H2",
    FuelSpec(
        name="Hydrogen",
        LHV_MJ_per_kg=120.0,
        O2_req_per_kg=8.000,                           # exact kg O2 / kg H2
        AFR_stoich=8.000 / O2_mass_fraction_in_air,    # ≈34.33
        MW_g_per_mol=2.016,
        rho_gas_kg_per_m3=0.084,
        flam_limits_volpct=(4.0, 75.0),
        S_L_300K_m_per_s=2.5,
        T_autoign_K=858.0,
        gamma_u_func=gamma_H2,
        phi_peak=1.0, phi_width=0.8, alpha_T=2.2, beta_p=-0.25, phi_minmax=(0.1, 3.0),
    ),
)
Fuels.register(
    "NH3",
    FuelSpec(
        name="Ammonia",
        LHV_MJ_per_kg=18.6,
        O2_req_per_kg=1.412,                            # kg O2 / kg NH3
        AFR_stoich=1.412 / O2_mass_fraction_in_air,     # ≈6.06
        MW_g_per_mol=17.031,
        rho_gas_kg_per_m3=0.73,
        flam_limits_volpct=(15.0, 28.0),
        S_L_300K_m_per_s=0.07,
        T_autoign_K=924.0,
        gamma_u_func=gamma_NH3,
        phi_peak=1.05, phi_width=0.25, alpha_T=2.0, beta_p=-0.25, phi_minmax=(0.7, 1.6),
    ),
)
Fuels.register(
    "CH4",
    FuelSpec(
        name="Methane",
        LHV_MJ_per_kg=50.0,
        O2_req_per_kg=17.2 * O2_mass_fraction_in_air,   # ≈4.01 kg O2 / kg fuel
        AFR_stoich=17.2,
        MW_g_per_mol=16.04,
        rho_gas_kg_per_m3=0.67,
        flam_limits_volpct=(5.0, 15.0),
        S_L_300K_m_per_s=0.38,
        T_autoign_K=813.0,
        gamma_u_func=lambda T: max(1.22, min(1.40, 1.39 - 1.0e-4 * (T - 300.0))),
        phi_peak=1.10, phi_width=0.35, alpha_T=2.0, beta_p=-0.25, phi_minmax=(0.6, 2.0),
    ),
)
#%% BLENDING
# Liquid densities [kg/L] for converting volume% to mass%
rho_liq = {
    "Gasoline": 0.74,
    "Ethanol":  0.789,
    "Methanol": 0.792,
}
def _volfrac_to_massfrac(vA: float, rhoA: float, rhoB: float) -> float:
    """
    Convert volume fraction vA (0..1) to mass fraction wA using liquid densities [kg/L].
    Suitable for oxygenate blending with gasoline.
    """
    vA = float(np.clip(vA, 0.0, 1.0))
    mA = vA * rhoA
    mB = (1.0 - vA) * rhoB
    return mA / (mA + mB + 1e-12)
def _blend_gamma_func(fuelA: FuelSpec, fuelB: FuelSpec, wA: float):
    wB = 1.0 - wA
    def g(T):
        return wA * fuelA.gamma_u(T) + wB * fuelB.gamma_u(T)
    return g
def _blend_linear(fuelA: FuelSpec, fuelB: FuelSpec, wA: float) -> Dict[str, float]:
    """
    Mass-fraction linear blend for LHV, O2_req, AFR_stoich, S_L reference.
    Return dict of key properties.
    """
    wB = 1.0 - wA
    LHV = wA*fuelA.LHV_MJ_per_kg + wB*fuelB.LHV_MJ_per_kg
    O2_req = wA*fuelA.O2_req_per_kg + wB*fuelB.O2_req_per_kg
    AFR = O2_req / O2_mass_fraction_in_air
    S0  = wA*fuelA.S_L_300K_m_per_s + wB*fuelB.S_L_300K_m_per_s
    return {"LHV_MJ_per_kg": LHV, "O2_req_per_kg": O2_req, "AFR_stoich": AFR, "S_L_300K_m_per_s": S0}
def blend_gasoline_ethanol(vE: float) -> FuelSpec:
    """
    Gasoline - Ethanol blend by VOLUME fraction of ethanol (0..1). Returns FuelSpec.
    """
    vE = float(np.clip(vE, 0.0, 1.0))
    Gas = Fuels.get("Gasoline")
    Eth = Fuels.get("Ethanol")
    wE  = _volfrac_to_massfrac(vE, rho_liq["Ethanol"], rho_liq["Gasoline"])
    props = _blend_linear(Eth, Gas, wE)   # mass blend (E first)
    return FuelSpec(
        name=f"E{int(round(vE*100)):02d} (Gasoline–Ethanol)",
        LHV_MJ_per_kg=props["LHV_MJ_per_kg"],
        O2_req_per_kg=props["O2_req_per_kg"],
        AFR_stoich=props["AFR_stoich"],
        MW_g_per_mol= (wE*Eth.MW_g_per_mol + (1-wE)*Gas.MW_g_per_mol),
        rho_gas_kg_per_m3= (wE*Eth.rho_gas_kg_per_m3 + (1-wE)*Gas.rho_gas_kg_per_m3),
        flam_limits_volpct=(min(Eth.flam_limits_volpct[0], Gas.flam_limits_volpct[0]),
                            max(Eth.flam_limits_volpct[1], Gas.flam_limits_volpct[1])),
        S_L_300K_m_per_s=props["S_L_300K_m_per_s"],
        T_autoign_K= wE*Eth.T_autoign_K + (1-wE)*Gas.T_autoign_K,
        gamma_u_func=_blend_gamma_func(Eth, Gas, wE),
        # carry over optional overrides by mass
        phi_peak = (wE*(Eth.phi_peak or 1.1) + (1-wE)*(Gas.phi_peak or 1.1)),
        phi_width= (wE*(Eth.phi_width or 0.35) + (1-wE)*(Gas.phi_width or 0.35)),
        alpha_T  = (wE*(Eth.alpha_T or 2.0)   + (1-wE)*(Gas.alpha_T or 2.0)),
        beta_p   = (wE*(Eth.beta_p or -0.25)  + (1-wE)*(Gas.beta_p or -0.25)),
        phi_minmax=None
    )
def blend_gasoline_methanol(vM: float) -> FuelSpec:
    """
    Gasoline - Methanol blend by VOLUME fraction of methanol (0..1). Returns FuelSpec.
    """
    vM = float(np.clip(vM, 0.0, 1.0))
    Gas = Fuels.get("Gasoline")
    Met = Fuels.get("Methanol")
    wM  = _volfrac_to_massfrac(vM, rho_liq["Methanol"], rho_liq["Gasoline"])
    props = _blend_linear(Met, Gas, wM)   # mass blend (M first)
    return FuelSpec(
        name=f"M{int(round(vM*100)):02d} (Gasoline–Methanol)",
        LHV_MJ_per_kg=props["LHV_MJ_per_kg"],
        O2_req_per_kg=props["O2_req_per_kg"],
        AFR_stoich=props["AFR_stoich"],
        MW_g_per_mol= (wM*Met.MW_g_per_mol + (1-wM)*Gas.MW_g_per_mol),
        rho_gas_kg_per_m3= (wM*Met.rho_gas_kg_per_m3 + (1-wM)*Gas.rho_gas_kg_per_m3),
        flam_limits_volpct=(min(Met.flam_limits_volpct[0], Gas.flam_limits_volpct[0]),
                            max(Met.flam_limits_volpct[1], Gas.flam_limits_volpct[1])),
        S_L_300K_m_per_s=props["S_L_300K_m_per_s"],
        T_autoign_K= wM*Met.T_autoign_K + (1-wM)*Gas.T_autoign_K,
        gamma_u_func=_blend_gamma_func(Met, Gas, wM),
        phi_peak = (wM*(Met.phi_peak or 1.1) + (1-wM)*(Gas.phi_peak or 1.1)),
        phi_width= (wM*(Met.phi_width or 0.35) + (1-wM)*(Gas.phi_width or 0.35)),
        alpha_T  = (wM*(Met.alpha_T or 2.0)   + (1-wM)*(Gas.alpha_T or 2.0)),
        beta_p   = (wM*(Met.beta_p or -0.25)  + (1-wM)*(Gas.beta_p or -0.25)),
        phi_minmax=None
    )
#  H2–NH3 BLENDING (mass / energy)
def blend_H2_NH3(w_H2: float) -> Dict[str, float]:
    """
    Blend H2 and NH3 by MASS fraction (0..1) and compute effective properties.
    Returns a dict of key parameters.
    """
    w_H2 = float(np.clip(w_H2, 0.0, 1.0))
    H2  = Fuels.get("H2")
    NH3 = Fuels.get("NH3")
    O2_req = w_H2 * H2.O2_req_per_kg + (1 - w_H2) * NH3.O2_req_per_kg
    AFR = O2_req / O2_mass_fraction_in_air
    LHV = w_H2 * H2.LHV_MJ_per_kg + (1 - w_H2) * NH3.LHV_MJ_per_kg
    SL  = (1 - w_H2) * NH3.S_L_300K_m_per_s + w_H2 * H2.S_L_300K_m_per_s
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
    Convert hydrogen ENERGY fraction (fraction of fuel energy from H2) -> MASS fraction of H2.
    """
    x_e_H2 = float(np.clip(x_e_H2, 0.0, 1.0))
    H2  = Fuels.get("H2")
    NH3 = Fuels.get("NH3")
    inv_e_H2  = 1.0 / H2.LHV_MJ_per_kg
    inv_e_NH3 = 1.0 / NH3.LHV_MJ_per_kg
    return (x_e_H2 * inv_e_H2) / (x_e_H2 * inv_e_H2 + (1 - x_e_H2) * inv_e_NH3)
def build_H2_NH3_blend_from_energy(x_e_H2: float) -> FuelSpec:
    """
    Build a FuelSpec for an H2–NH3 blend from an ENERGY fraction of H2 (0..1).
    """
    x_e_H2 = float(np.clip(x_e_H2, 0.0, 1.0))
    w_H2 = mass_fraction_from_energy_fraction(x_e_H2)
    mix = blend_H2_NH3(w_H2)
    H2  = Fuels.get("H2"); NH3 = Fuels.get("NH3")
    return FuelSpec(
        name=f"H2–NH3 ({int(round(x_e_H2*100))}% H2 energy)",
        LHV_MJ_per_kg=mix["LHV_MJ_per_kg"],
        O2_req_per_kg=mix["O2_req_per_kg"],
        AFR_stoich=mix["AFR_stoich"],
        MW_g_per_mol= w_H2*H2.MW_g_per_mol + (1-w_H2)*NH3.MW_g_per_mol,
        rho_gas_kg_per_m3= w_H2*H2.rho_gas_kg_per_m3 + (1-w_H2)*NH3.rho_gas_kg_per_m3,
        flam_limits_volpct=(min(H2.flam_limits_volpct[0], NH3.flam_limits_volpct[0]),
                            max(H2.flam_limits_volpct[1], NH3.flam_limits_volpct[1])),
        S_L_300K_m_per_s=mix["S_L_300K_m_per_s"],
        T_autoign_K= w_H2*H2.T_autoign_K + (1-w_H2)*NH3.T_autoign_K,
        gamma_u_func=lambda T: (w_H2*H2.gamma_u(T) + (1-w_H2)*NH3.gamma_u(T)),
        # wide φ range thanks to H2; feel free to tune
        phi_peak=1.0, phi_width=0.6, alpha_T=2.1, beta_p=-0.25, phi_minmax=(0.3, 2.2),
    )
# EMISSIONS: CO2 EMISSION INDEX (g/kg FUEL) HELPER
def ei_co2_g_per_kg_for(fuel: FuelSpec) -> float:
    """
    Very rough engine-out CO2 EI by fuel. Use to scale CO2 = EI * m_dot_fuel.
    H2 and NH3 ≈ 0 g/kg fuel (carbon-free); CH4 lower than gasoline; alcohols in-between.
    """
    name = fuel.name.lower()
    if "hydrogen" in name or "h2" in name:
        return 0.0
    if "ammonia" in name or "nh3" in name:
        return 0.0
    if "methane" in name or "ch4" in name or "natural gas" in name:
        return 2850.0
    if "methanol" in name:
        return 1370.0
    if "ethanol" in name or "e" in name: # crude: neat ethanol ~1910; blends handled elsewhere typically
        return 1910.0
    if "gasoline" in name:
        return 3090.0
    return 3000.0 # generic fallback