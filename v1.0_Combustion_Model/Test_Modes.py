from engine_model import  combustion_Wiebe, estimate_Emissions
import Calibration as cal
from Engine_Database import EngineSpec
from typing import Optional, Iterable
import numpy as np
import pandas as pd
def RunPoint(spec: EngineSpec, rpm: int, throttle: float, analyze: bool = False, combustion_kwargs: Optional[dict] = None) -> dict:
    """
    Single operating point
    """
    if combustion_kwargs is None:
        combustion_kwargs = {}
    # VE from table
    table = getattr(spec, "ve_table", None)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"spec.ve_table must be a pandas DataFrame, got {type(table)}")
    ve_val = cal.get_ve_from_table(rpm, throttle, table)
    ve = float(np.asarray(ve_val).ravel()[0])
    # COMBUSTION
    res = combustion_Wiebe(spec = spec, rpm = rpm, throttle = throttle, ve = ve, plot = False, return_dic = True)
    if not isinstance(res, dict):
        raise RuntimeError("combustion_Wiebe did not return a dict. Ensure return_dic=True is honored.")
    # --- per-cycle -> per-second (4-stroke) ---
    cps = rpm / 120.0  # cycles per second per cylinder
    mdot_air  = res["m_air_per_cycle"]  * cps * spec.n_cylinder  # kg/s
    mdot_fuel = res["m_fuel_per_cycle"] * cps * spec.n_cylinder  # kg/s
    # --- emissions & BSFC ---
    AFR = cal.get_target_AFR(rpm)
    CO2_gps, CO_gps, NOx_gps, HC_gps = estimate_Emissions(mdot_fuel, AFR, 0.98)
    PkW = max(res["power_kw"], 1e-9)
    BSFC_g_per_kWh = (mdot_fuel * 3600.0) / PkW
    to_gkWh = lambda gps: (gps * 3600.0) / PkW
    return {
        "RPM": rpm, "Throttle": throttle, "VE": ve,
        "Torque_Nm": res["torque_nm"], "Power_kW": res["power_kw"],
        "IMEP_bar": res["imep_gross_pa"]/1e5, "BMEP_bar": res["bmep_pa"]/1e5,
        "FMEP_bar": res["fmep_pa"]/1e5, "PMEP_bar": res["pmep_pa"]/1e5,
        "mdot_air_kg_s": mdot_air, "mdot_fuel_kg_s": mdot_fuel,
        "BSFC_g_per_kWh": BSFC_g_per_kWh,
        "CO2_gps": CO2_gps, "CO_gps": CO_gps, "NOx_gps": NOx_gps, "HC_gps": HC_gps,
        "CO2_g_kWh": to_gkWh(CO2_gps), "CO_g_kWh": to_gkWh(CO_gps),
        "NOx_g_kWh": to_gkWh(NOx_gps), "HC_g_kWh": to_gkWh(HC_gps),
        "CA10_deg": res["ca10_deg"], "CA50_deg": res["ca50_deg"], "CA90_deg": res["ca90_deg"],
        "Pmax_bar": res["pmax_bar"], "Tmax_K":  res["tmax_k"],
    }
def SingleRun(spec, rpm: int, throttles: Optional[Iterable[float]] = None, analyze_throttles: Optional[Iterable[float]] = None, combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    For a single engine speed, sweep throttle (e.g., 0.1..1.0) using VE from spec.ve_table.
    Returns a DataFrame with one row per (rpm, throttle).
    """
    if throttles is None:
        throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    analyze_set = set(analyze_throttles or [])
    rows = []
    for th in throttles:
        rows.append(
            RunPoint(spec=spec,
                     rpm=int(rpm),
                     throttle=float(th),
                     analyze=(th in analyze_set),
                     combustion_kwargs=combustion_kwargs)
        )
    return pd.DataFrame(rows).sort_values(["RPM","Throttle"]).reset_index(drop=True)
def WideOpenThrottle(spec, RPM_min: int, RPM_max: int, step: int = 100, *, analyze_points: Optional[Iterable[int]] = None, combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    WOT sweep using VE from spec.ve_table. Returns one row per RPM.
    """
    analyze_set = set(analyze_points or [])
    rows = []
    for r in range(int(RPM_min), int(RPM_max)+1, int(step)):
        rows.append(
            RunPoint(spec=spec,
                     rpm=int(r),
                     throttle=1.0,
                     analyze=(r in analyze_set),
                     combustion_kwargs=combustion_kwargs)
        )
    return pd.DataFrame(rows).sort_values("RPM").reset_index(drop=True)
def FullRangeSweep(spec, RPM_min: int, RPM_max: int, step: int = 100, throttles: Optional[Iterable[float]] = None, *, combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    Grid sweep over RPM and throttle using VE from spec.ve_table.
    Returns a DataFrame with one row per (RPM,Throttle).
    """
    if throttles is None:
        throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for r in range(int(RPM_min), int(RPM_max)+1, int(step)):
        for th in throttles:
            rows.append(
                RunPoint(spec=spec,
                         rpm=int(r),
                         throttle=float(th),
                         analyze=False,
                         combustion_kwargs=combustion_kwargs)
            )
    return pd.DataFrame(rows).sort_values(["RPM","Throttle"]).reset_index(drop=True)
def DesignComparison():
    return
