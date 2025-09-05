import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    mdotFuel = mdotAir / cal.get_target_AFR(rpm)
    gross_torque = mdotFuel * LHV * eff / (rpm * 2 * np.pi / 60)
    fmep = 0.25 + 0.02 * rpm / 1000 + 0.03 * (rpm / 1000) ** 2  # bar
    fmep_pa = fmep * 1e5
    displacement_m3 = displacement_l / 1e3
    torque_fmep = fmep_pa * displacement_m3 / (4 * np.pi)
    pmep = 0.02 + 0.00001 * rpm  # bar
    pmep_pa = pmep * 1e5
    torque_pmep = pmep_pa * displacement_m3 / (4 * np.pi)
    torque_net = gross_torque - (torque_fmep + torque_pmep)
    emissions = estimate_Emissions(mdotFuel, cal.get_target_AFR(rpm), eff)
    return max(torque_net, 0), mdotFuel, emissions
def calculate_power(rpm, torque):
    '''
    Convert torque (Nm) and engine speed (rpm) into power output (kW).

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        torque (float): Net brake torque in Nm.

    Returns:
        float: Power in kilowatts (kW).
    '''
    return (torque * rpm * 2 * np.pi / 60) / 1000
def calculate_horsePower(rpm, torque):
    '''
    Convert torque (Nm) and engine speed (rpm) into power output in horsepower.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        torque (float): Net brake torque in Nm.

    Returns:
        float: Power in mechanical horsepower (hp).
    '''
    return (torque * rpm * 2 * np.pi / 60) / 745.7
def estimate_Emissions(mDotFuel, AFR, eff):
    """
    Rough estimate of emissions based on fuel mass flow and AFR.
    fuel_mass_flow: kg/s
    afr: actual AFR
    combustion_eff: assumed combustion efficiency (0-1)
    Returns CO2, CO, NOx and HC in g/s
    """
    # Empirical Scaling Factors
    k_co = 0.2
    k_nox = 0.08
    k_thc = 0.03
    mDotFuel = mDotFuel * 1000
    lambda_val = AFR / 14.7
    mDotco2 = mDotFuel * 3.09 # CO2: 3.09 g CO2 per g fuel burned
    mDotco = mDotFuel * k_co * max(0, abs(1.2 - lambda_val)) # CO: Rises when rich (lambda < 1.2)
    mDotnox = mDotFuel * k_nox * max(0, 1 - abs(lambda_val - 1)) # NOx: peaks near stoichiometric, lower when far rich/lean
    mDotthc = mDotFuel * k_thc * (1 - eff) # THC mainly from incomplete combustion
    emissions_gps = [mDotco2, mDotco, mDotnox, mDotthc] # g/s
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
    EI_NOx = peak_noX * np.exp(-0.5*((lam - lam_peak)/sigma)**2)
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
def combustion_Wiebe(displacement_l =3.50, n_cylinder = 4, bore = 0.086, stroke = 0.086, 
                     conrod = 0.143, compressionRatio = 10, rpm = 300, throttle = 1.0, ve = 0.9, 
                     LHV = 44E6, rho = 1.22588, gas_constant = 287, T_ivc = 330, 
                     a = 5, m = 2, combustion_efficiency = 0.98, c_v = 1.14):
    # GEOMETRY
    V_displacement = np.pi * (bore**2 / 4) * stroke
    V_clearance = V_displacement / (compressionRatio - 1)
    crank_radius = stroke / 2
    crossSec = np.pi * bore**2 / 4
    # POSITION
    crank_angle = np.linspace(-np.pi, np.pi, 1441)
    piston_pos = crank_radius * (1 - np.cos(crank_angle)) + crank_radius**2 / (2 * conrod) * (1 - np.cos(2 * crank_angle))
    V = V_clearance + crossSec * piston_pos
    dV_dtheta = np.gradient(V, crank_angle) # Numerical derivative dV/dθ
    # TIMING
    ivc_rad = np.deg2rad(-110.0) # Approximate
    soc_rad = np.deg2rad(-10.0) # Approximate
    eoc_rad = np.deg2rad(25.0) # Approximate
    i_ivc = int(np.argmin(np.abs(crank_angle - ivc_rad)))
    i_soc = int(np.argmin(np.abs(crank_angle - soc_rad)))
    i_eoc = int(np.argmin(np.abs(crank_angle - eoc_rad)))
    # TRAPPED MASS
    p_ivc = (20 + throttle * (100 - 20)) * 1e3
    m_trapped = p_ivc * V[i_ivc] / (gas_constant * T_ivc)
    # COMPRESSION STROKE
    V_compression = V[i_ivc:i_soc+1]
    P_compression = (p_ivc * (V[i_ivc]/ V_compression) ** 1.34) # Polytropic compression
    T_compression = (P_compression * V_compression / (m_trapped * gas_constant))
    # COMBUSTION
    delta = eoc_rad - soc_rad
    P_current = P_compression [-1]
    T_current = T_compression [-1]
    P_combustion = []
    T_combustion = []
    mfb_list = []

    mdotair = (calculate_air_mass_flow(rpm, displacement_l, ve) / n_cylinder) / (rpm / 120.0)
    mdotfuel = mdotair / cal.get_target_AFR(rpm)
    Q_tot = mdotfuel * combustion_efficiency * LHV
    dtheta = crank_angle[1] - crank_angle[0]
    
    for i in range(i_soc, i_eoc+1):
        theta = crank_angle[i]
        x = (theta - soc_rad) / delta
        x = np.clip(x, 0.0, 1.0)
        mfb = 1.0 - np.exp(-a * x**(m+1))
        dmfb_dtheta = a * (m+1) * x**m * np.exp(-a * x**(m+1)) / delta
        dQ = Q_tot * dmfb_dtheta
        dQ_step = dQ * dtheta
        dT_combustion = dQ_step / (m_trapped * c_v)
        V_current = V[i +1]
        T_current = T_current + dT_combustion
        P_current = (m_trapped * gas_constant * T_current) / V_current
        T_combustion.append(T_current)
        P_combustion.append(P_current)
        mfb_list.append(mfb)
    df = pd.DataFrame({
        'Crank Angle (deg)': np.degrees(crank_angle),
        'Volume (m3)': V,
        'dVdtheta (m3/rad)': dV_dtheta,
        'Pressure (bar)': np.nan,
        'Temperature (K)':  np.nan,
        'Mass Fraction Burned': np.nan
    })
    df.loc[i_ivc:i_soc, 'Pressure (bar)'] = P_compression / 1e5
    df.loc[i_soc:i_eoc + 1, 'Pressure (bar)'] = np.array(P_combustion / 1e5)
    df.loc[i_ivc:i_soc, 'Temperature (K)']  = T_compression
    df.loc[i_soc:i_eoc + 1, 'Temperature (K)'] =np.array(T_combustion)
    df.loc[i_soc:i_eoc + 1, 'Mass Fraction Burned'] = np.array(mfb_list)
    return
