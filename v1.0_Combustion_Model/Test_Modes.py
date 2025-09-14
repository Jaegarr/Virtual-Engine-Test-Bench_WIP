from engine_model import calculate_air_mass_flow, calculate_power, calculate_torque, calculate_horsePower, combustion_Wiebe, estimate_Emissions
import Calibration as cal
from Engine_Database import EngineSpec, Engines
from typing import Optional
import numpy as np
import pandas as pd
def RunPoint(spec: EngineSpec, rpm: int, throttle: float, constant_ve: float, ve_mode: str = 'constant', ve_table=None, analyze: bool = False, combustion_kwargs: Optional[dict] = None) -> dict: 
    if combustion_kwargs is None:
        combustion_kwargs = {}
        if ve_mode == 'table':
            ve_val = cal.get_ve_from_table(rpm, throttle, ve_table)
        else:
            ve_val = [constant_ve]
        ve = float(np.asarray(ve_val).ravel()[0])
        result = combustion_Wiebe(spec=spec, rpm=rpm, throttle=throttle, ve=ve, plot=False, return_dic=True)
    if result is None:
        raise RuntimeError("combustion_Wiebe did not return a dict. Ensure return_dic=True is honored inside the function.")
    # 
    cycles_per_sec_per_cyl = rpm / 120.0
    mdot_air  = result["m_air_per_cycle"]  * cycles_per_sec_per_cyl * spec.n_cylinder
    mdot_fuel = result["m_fuel_per_cycle"] * cycles_per_sec_per_cyl * spec.n_cylinder
    # 
    AFR = cal.get_target_AFR(rpm)
    CO2_gps, CO_gps, NOx_gps, HC_gps = estimate_Emissions(mdot_fuel, AFR, 0.98)
    #  
    PkW = max(result["power_kw"], 1e-9)
    to_gkWh = lambda gps: (gps * 3600.0) / PkW
    BSFC_g_per_kWh = (mdot_fuel * 1000.0 * 3600.0) / PkW
    # 
    out = {
        "RPM": rpm,
        "Throttle": throttle,
        "VE": ve,
        "Torque (Nm)": result["torque_nm"],
        "Power (kW)":  result["power_kw"],
        "IMEP (bar)":  result["imep_gross_pa"]/1e5,
        "BMEP (bar)":  result["bmep_pa"]/1e5,
        "FMEP (bar)":  result["fmep_pa"]/1e5,
        "PMEP (bar)":  result["pmep_pa"]/1e5,
        "Air Flow (g/s)":  mdot_air,
        "Fuel Flow(g/s)": mdot_fuel,
        "BSFC (g/kWh)": BSFC_g_per_kWh,
        "CO2_gps": CO2_gps, 
        "CO_gps": CO_gps, 
        "NOx_gps": NOx_gps, 
        "HC_gps": HC_gps,
        "CO2_g_kWh": to_gkWh(CO2_gps), 
        "CO_g_kWh": to_gkWh(CO_gps),
        "NOx_g_kWh": to_gkWh(NOx_gps), 
        "HC_g_kWh": to_gkWh(HC_gps),
        "CA10_deg": result["ca10_deg"], 
        "CA50_deg": result["ca50_deg"], 
        "CA90_deg": result["ca90_deg"],
        "Pmax_bar": result["pmax_bar"], 
        "Tmax_K": result["tmax_k"],
    }
    return out
    return
def SingleRun(rpm, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Perform a single engine speed simulation across a range of throttle positions.

    Parameters:
        rpm (int or float): Engine speed in revolutions per minute.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle, Torque (Nm), Power (kW), Horsepower].
                       The list covers throttle values from 0.1 to 1.0 in 0.1 increments.
    """
    results = []
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if ve_mode == 'table':
        ve_list = cal.get_ve_from_table(rpm, throttles, ve_table)
    else:
        ve_list = [constant_ve] * len(throttles)
    for throttle, ve in zip(throttles, ve_list):
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t, mdotFuel, emissions = calculate_torque(rpm, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm, t)
        row = [rpm, throttle, t, p, hp, mdotAir, mdotFuel] + emissions
        results.append(row)
    return results
def WideOpenThrottle(spec,RPM_min: int, RPM_max: int, step: int = 100, *, ve_mode: str = 'constant', constant_ve: float = 0.98, ve_table=None, analyze_points: list[int] | None = None, combustion_kwargs: dict | None = None) -> pd.DataFrame:
    """
    WOT sweep (Throttle = 1.0) from RPM_min to RPM_max.
    Uses RunPoint(...) at each RPM and returns a DataFrame of results.

    Parameters
    ----------
    spec : EngineSpec
        Engine definition from your DB (or custom).
    RPM_min, RPM_max : int
        Sweep range, inclusive.
    step : int, default 100
        RPM step.
    ve_mode : {'constant','table'}, default 'constant'
        How to resolve VE at each point.
    constant_ve : float, default 0.98
        VE used when ve_mode='constant'.
    ve_table : any
        Passed through to your VE interpolation when ve_mode='table'.
    analyze_points : list[int] | None
        RPMs at which to enable comb. plots (RunPoint analyze=True).
    combustion_kwargs : dict | None
        Optional knobs passed into combustion_Wiebe (e.g., a, m, etc.).

    Returns
    -------
    pd.DataFrame
        Rows per RPM with columns from RunPoint (Torque_Nm, Power_kW, etc.).
    """
    if combustion_kwargs is None:
        combustion_kwargs = {}
    if analyze_points is None:
        analyze_points = []
    analyze_set = set(analyze_points)
    rows = []
    for rpm in range(int(RPM_min), int(RPM_max) + 1, int(step)):
        row = RunPoint(spec=spec, rpm=rpm,throttle=1.0, ve_mode=ve_mode, constant_ve=constant_ve, ve_table=ve_table, analyze=(rpm in analyze_set), combustion_kwargs=combustion_kwargs)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("RPM").reset_index(drop=True)
    return df
def FullRangeSweep(RPM_min, RPM_max, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Perform a full simulation sweep across RPM and throttle range.

    Parameters:
        RPM_min (int): Minimum RPM for the sweep.
        RPM_max (int): Maximum RPM for the sweep.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle, Torque (Nm), Power (kW), Horsepower].
                       Covers RPM range in steps of 100 RPM and throttle from 0.1 to 1.0 in 0.1 increments.
    """
    results = []
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for rpm in range(RPM_min, RPM_max + 1, 100):
        if ve_mode == 'table':
            ve_list = cal.get_ve_from_table(rpm, throttles, ve_table)
        else:
            ve_list = [constant_ve] * len(throttles)
        for throttle, ve in zip(throttles, ve_list):
            mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
            t, mdotFuel, emissions = calculate_torque(rpm, mdotAir, displacement_l)
            p = calculate_power(rpm, t)
            hp = calculate_horsePower(rpm, t)
            row = [rpm, throttle, t, p, hp, mdotAir, mdotFuel] + emissions
            results.append(row)
    return results
def DesignComparison():
    return

if __name__ == "__main__":
    spec = Engines.get("Nissan_VQ35DE__NA_3.5L_V6_350Z")

    pt = RunPoint(
        spec=spec,
        rpm=3000,
        throttle=1.0,
        constant_ve=0.98,
        ve_table='constant',
        analyze=True,                   
    )

    print(f"Torque: {pt['Torque (Nm)']:.1f} Nm | Power: {pt['Power (kW)']:.1f} kW")
    print(f"IMEP: {pt['IMEP (bar)']:.2f} bar | BMEP: {pt['BMEP (bar)']:.2f} bar")
    print(f"Pmax: {pt['Pmax_bar']:.1f} bar | Tmax: {pt['Tmax_K']:.0f} K")
    print(f"BSFC: {pt['BSFC (g/kWh)']:.0f} g/kWh")