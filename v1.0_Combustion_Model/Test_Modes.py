from engine_model import calculate_air_mass_flow, calculate_power, calculate_torque, calculate_horsePower, combustion_Wiebe, estimate_Emissions
import Calibration as cal
from Engine_Database import EngineSpec, Engines
from typing import Optional
def RunPoint(spec: EngineSpec, rpm: int, throttle: float, constant_ve: float, ve_mode: str = 'constant', ve_table=None, analyze: bool = False, combustion_kwargs: Optional[dict] = None) -> dict: 
    if combustion_kwargs is None:
        combustion_kwargs = {}
        if ve_mode == 'table':
            ve = cal.get_ve_from_table(rpm, throttle, ve_table)
        else:
            ve = [constant_ve]
        res = combustion_Wiebe(spec=spec, rpm=rpm, throttle=throttle, ve=ve, plot=analyze, return_dic=True)
    if res is None:
        raise RuntimeError("combustion_Wiebe did not return a dict. Ensure return_dic=True is honored inside the function.")
    # 3) Per-cycle -> mass flows [kg/s]  (4-stroke: rpm/120 cycles per cylinder per second)
    cycles_per_sec_per_cyl = rpm / 120.0
    mdot_air  = res["m_air_per_cycle"]  * cycles_per_sec_per_cyl * spec.n_cylinder
    mdot_fuel = res["m_fuel_per_cycle"] * cycles_per_sec_per_cyl * spec.n_cylinder

    # 4) Emissions (your model expects kg/s fuel, returns g/s)
    AFR = cal.get_target_AFR(rpm)
    CO2_gps, CO_gps, NOx_gps, HC_gps = estimate_Emissions(mdot_fuel, AFR, 0.98)

    # 5) Intensities (guard power)
    PkW = max(res["power_kw"], 1e-9)
    to_gkWh = lambda gps: (gps * 3600.0) / PkW
    BSFC_g_per_kWh = (mdot_fuel * 3600.0) / PkW

    # 6) Package a clean row
    out = {
        "RPM": rpm,
        "Throttle": throttle,
        "VE": ve,
        "Torque (Nm)": res["torque_nm"],
        "Power (kW)":  res["power_kw"],
        "IMEP (bar)":  res["imep_gross_pa"]/1e5,
        "BMEP (bar)":  res["bmep_pa"]/1e5,
        "FMEP (bar)":  res["fmep_pa"]/1e5,
        "PMEP (bar)":  res["pmep_pa"]/1e5,

        "Air Flow (g/s)":  mdot_air,
        "Fuel Flow(g/s)": mdot_fuel,

        "BSFC (g/kWh)": BSFC_g_per_kWh,

        "CO2_gps": CO2_gps, "CO_gps": CO_gps, "NOx_gps": NOx_gps, "HC_gps": HC_gps,
        "CO2_g_kWh": to_gkWh(CO2_gps), "CO_g_kWh": to_gkWh(CO_gps),
        "NOx_g_kWh": to_gkWh(NOx_gps), "HC_g_kWh": to_gkWh(HC_gps),

        "CA10_deg": res["ca10_deg"], "CA50_deg": res["ca50_deg"], "CA90_deg": res["ca90_deg"],
        "Pmax_bar": res["pmax_bar"], "Tmax_K": res["tmax_k"],
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
def WideOpenThrottle(RPM_min, RPM_max, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Simulate engine performance at full throttle (100%) over a specified RPM range.

    Parameters:
        RPM_min (int): Minimum RPM for the simulation.
        RPM_max (int): Maximum RPM for the simulation.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle=1.0, Torque (Nm), Power (kW), Horsepower].
                       Covers RPM range in steps of 100 RPM.
    """
    results = []
    throttle = 1.0
    for rpm in range(RPM_min, RPM_max + 1, 100):
        if ve_mode == 'table':
            ve = cal.get_ve_from_table(rpm, throttle, ve_table)
        else:
            ve = constant_ve
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t, mdotFuel, emissions = calculate_torque(rpm, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm, t)
        row = [rpm, throttle, t, p, hp, mdotAir*1000, mdotFuel*1000] + emissions
        results.append(row)
    return results
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

    print(f"Torque: {pt['Torque_Nm']:.1f} Nm | Power: {pt['Power_kW']:.1f} kW")
    print(f"IMEP: {pt['IMEP_bar']:.2f} bar | BMEP: {pt['BMEP_bar']:.2f} bar")
    print(f"Pmax: {pt['Pmax_bar']:.1f} bar | Tmax: {pt['Tmax_K']:.0f} K")
    print(f"BSFC: {pt['BSFC_g_per_kWh']:.0f} g/kWh")