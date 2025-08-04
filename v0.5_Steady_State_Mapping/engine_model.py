import math
import Calibration as cal
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

