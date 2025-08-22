import math
import Calibration as cal
from scipy.interpolate import RegularGridInterpolator
def calculate_air_mass_flow(rpm, displacement_l, ve, rho=1.22588):
    '''
    Estimate the air mass flow rate (kg/s) into a naturally aspirated 4‑stroke engine.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        displacement_l (float): Engine displacement in liters.
        ve (float): Volumetric efficiency (typically 0.6–1.2).
        rho (float): Air density in kg/m³ (default is 1.22588 for standard conditions).

    Returns:
        float: Air mass flow rate in kg/s.

    Notes:
        - Assumes a naturally aspirated 4‑stroke engine (hence division by 2).
        - Simplified steady‑state calculation; does not account for transient or forced induction effects.
    '''
    displacement_m3 = displacement_l / 1e3
    mdot = displacement_m3 * ve * rpm * rho / (2 * 60)
    return mdot
def calculate_torque(rpm, mdotAir, displacement_l, LHV=44e6, eff=0.3):
    '''
    Estimate engine brake torque (Nm) based on air mass flow and engine parameters.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        mdotAir (float): Air mass flow rate in kg/s.
        displacement_l (float): Engine displacement in liters.
        LHV (float): Lower heating value of fuel in J/kg (default: 44 MJ/kg).
        eff (float): Brake thermal efficiency (default: 0.3).

    Returns:
        float: Net brake torque in Nm (≥0).

    Notes:
        - Uses target lambda from calibration map to estimate fuel mass flow.
        - Subtracts simplified frictional (FMEP) and pumping (PMEP) losses based on empirical formulas.
    '''
    mdotFuel = mdotAir / cal.get_target_lambda(rpm)
    gross_torque = mdotFuel * LHV * eff / (rpm * 2 * math.pi / 60)
    fmep = 0.25 + 0.02 * rpm / 1000 + 0.03 * (rpm / 1000) ** 2  # bar
    fmep_pa = fmep * 1e5
    displacement_m3 = displacement_l / 1e3
    torque_fmep = fmep_pa * displacement_m3 / (4 * math.pi)
    pmep = 0.02 + 0.00001 * rpm  # bar
    pmep_pa = pmep * 1e5
    torque_pmep = pmep_pa * displacement_m3 / (4 * math.pi)
    torque_net = gross_torque - (torque_fmep + torque_pmep)
    emissions = estimate_Emissions(mdotFuel, cal.get_target_lambda(rpm), eff)
    return max(torque_net, 0), emissions
def calculate_power(rpm, torque):
    '''
    Convert torque (Nm) and engine speed (rpm) into power output (kW).

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        torque (float): Net brake torque in Nm.

    Returns:
        float: Power in kilowatts (kW).
    '''
    return (torque * rpm * 2 * math.pi / 60) / 1000
def calculate_horsePower(rpm, torque):
    '''
    Convert torque (Nm) and engine speed (rpm) into power output in horsepower.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        torque (float): Net brake torque in Nm.

    Returns:
        float: Power in mechanical horsepower (hp).
    '''
    return (torque * rpm * 2 * math.pi / 60) / 745.7
def estimate_Emissions(mDotFuel, AFR, eff):
    """
    Rough estimate of emissions based on fuel mass flow and AFR.
    fuel_mass_flow: kg/s
    afr: actual AFR
    combustion_eff: assumed combustion efficiency (0-1)
    Returns CO2, CO, NOx and HC in g/s
    """
    # Empirical Scaling Factors
    k_co = 200
    k_nox = 80
    k_thc = 30
    lambda_val = AFR / 14.7
    mDotco2 = mDotFuel * 3.09 # CO2: 3.09 kg CO2 per kg fuel burned
    mDotco = mDotFuel * k_co * max(0, abs(1.2 - lambda_val)) # CO: Rises when rich (lambda < 1.2)
    mDotnox = mDotFuel * k_nox * max(0, 1 - abs(lambda_val - 1)) # NOx: peaks near stoichiometric, lower when far rich/lean
    mDotthc = mDotFuel * k_thc * (1 - eff) # THC mainly from incomplete combustion
    emissions_kgps = [mDotco2, mDotco, mDotnox, mDotthc] # g/s
    emissions_gps = [val * 1000 for val in emissions_kgps]
    return emissions_gps

def estimate_emissions(mDotFuel, AFR, comb_eff, load_frac=0.6, ei_co2_g_per_kg=3090.0):
    """
    Estimate engine-out emissions as g/s from fuel flow, AFR, and a load proxy.

    Parameters
    ----------
    mDotFuel_kgps : float
        Fuel mass flow [kg/s]
    AFR : float
        Actual air-fuel ratio (mass-based). Stoich gasoline ≈ 14.7.
    comb_eff : float
        Combustion efficiency (0..1). Influences HC primarily.
    load_frac : float
        0..1 proxy for load/BMEP (affects NOx amplitude).
    ei_co2_g_per_kg : float
        Emission index for CO2 [g/kg fuel]. ~3090 for gasoline.

    Returns
    -------
    dict : {'CO2': g/s, 'CO': g/s, 'NOx': g/s, 'HC': g/s}
        Engine-out (pre-catalyst) emission rates.
    """

    lam = AFR / 14.7
    lam = max(0.5, min(1.6, lam))
    # --- CO (g/kg fuel) ---
    # Very low when lean; rises rapidly rich of stoich.
    # Smooth curve: quadratic increase as lambda goes below 1.
    # Typical hot engine-out EI_CO at λ≈0.9 can be O(100 g/kg), lean ~<5 g/kg.
    if lam >= 1.0:
        EI_CO = 3.0 * (1 + 4.0*(lam - 1.0))  # slightly increases if very lean due to misfire risk
    else:
        EI_CO = 5.0 + 800.0*(1.0 - lam)**2   # rich penalty
    EI_CO = min(EI_CO, 400.0)  # cap to avoid extremes

    # --- HC (g/kg fuel) ---
    # Rises rich (over-fuel/quench) and very lean (misfire), plus incomplete combustion.
    rich_term = 200.0*(max(0.0, 1.0 - lam))**2
    lean_term = 120.0*(max(0.0, lam - 1.15))**2
    incomp_term = 50.0*(1.0 - max(0.0, min(1.0, comb_eff)))
    EI_HC = 2.0 + rich_term + lean_term + incomp_term
    EI_HC = min(EI_HC, 300.0)

    # --- NOx (g/kg fuel) ---
    # Peak slightly lean of stoich; scale with load (temperature).
    # Use a Gaussian around lambda≈1.05 with width ~0.08–0.10.
    lam_peak = 1.05
    sigma = 0.09
    peak_noX = 18.0 * (load_frac**0.7)  # higher load → more NOx
    EI_NOx = peak_noX * math.exp(-0.5*((lam - lam_peak)/sigma)**2)
    # Mild lean/rich suppression already handled by Gaussian

    # --- CO2 (g/kg fuel) ---
    EI_CO2 = ei_co2_g_per_kg  # essentially fixed by fuel chemistry

    # Convert EI [g/kg fuel] to g/s using fuel flow [kg/s]
    gps_CO2 = EI_CO2 * mDotFuel
    gps_CO  = EI_CO  * mDotFuel
    gps_NOx = EI_NOx * mDotFuel
    gps_HC  = EI_HC  * mDotFuel

    return {'CO2': gps_CO2, 'CO': gps_CO, 'NOx': gps_NOx, 'HC': gps_HC}
'''
def get_bsfc_from_table(torque, rpm, BSFC_table):
    bsfc_results = []
    torque_values = BSFC_table.index.to_numpy(dtype=float)
    rpm_values = BSFC_table.columns.to_numpy(dtype=float)
    bsfc_values = BSFC_table.to_numpy()
    bsfc = RegularGridInterpolator((torque_values, rpm_values), bsfc_values, bounds_error=False, fill_value=None)
    return bsfc
'''
