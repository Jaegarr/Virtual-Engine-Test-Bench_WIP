import torch
import json
import pandas as pd
import Calibration as cal
from Calibration import lambda_target_map
from typing import Optional, Iterable
from engine_model import  combustion_Wiebe, estimate_Emissions
from Engine_Database import EngineSpec
from Fuel_Database import FuelSpec
from ML_Correction import MLPCorrection

USE_ML_CORRECTION = False # MAGIC BUTTON
def load_correction_model(model_path, meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    # meta key fix + safe default
    model = MLPCorrection(with_vvl_flag=meta.get("use_vvl_flag", True))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
ml_model = load_correction_model(
    r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\mlp_correction.pt",
    r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\mlp_correction_meta.json")
def RunPoint(spec: EngineSpec, 
             fuel: FuelSpec,  
             rpm: int, 
             throttle: float, 
             analyze: bool = False, 
             combustion_kwargs: Optional[dict] = None) -> dict:

    combustion_kwargs = dict(combustion_kwargs or {})

    # --- VE from table ---
    table = getattr(spec, "ve_table", None)
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"spec.ve_table must be a pandas DataFrame, got {type(table)}")
    ve = float(cal.get_ve_from_table(rpm, throttle, table))

    # --- ML corrections (ΔFMEP [bar], ΔSOC [deg], Δλ [–]) ---
    fmep_offset_Pa = 0.0
    soc_offset_deg = 0.0
    d_lambda       = 0.0

    if USE_ML_CORRECTION == True and (ml_model is not None):
        
        print("MLP loaded:", any(p.requires_grad for p in ml_model.parameters()))
        for test_rpm in [1000, 2000, 3000, 4000, 5000, 6000, 7000]:
            vvl_flag = 1.0 if test_rpm >= 4500 else 0.0
            out = ml_model(test_rpm, vvl_flag)
            if isinstance(out, (tuple, list)) and len(out) == 3:
                d_fmep_bar, d_soc_deg, d_lam = out
            else: 
                d_fmep_bar, d_soc_deg = out
                d_lam = torch.tensor(0.0)
            dfmep = float(d_fmep_bar.detach().cpu().numpy())
            dsoc  = float(d_soc_deg.detach().cpu().numpy())
            dlam  = float(d_lam.detach().cpu().numpy())
            print(f"{test_rpm:>5} RPM : Δλ={dlam:+.4f}  ΔSOC={dsoc:+.3f}°CA  ΔFMEP={dfmep:+.4f} bar")
        # actual offsets for this operating point
        vvl_flag = 1.0 if rpm >= 4500 else 0.0
        out = ml_model(rpm, vvl_flag)
        if isinstance(out, (tuple, list)) and len(out) == 3:
            d_fmep_bar, d_soc_deg, d_lam = out
        else:
            d_fmep_bar, d_soc_deg = out
            d_lam = torch.tensor(0.0)
        fmep_offset_Pa = float(d_fmep_bar.detach().cpu().numpy()) * 1e5  # Pa
        soc_offset_deg = float(d_soc_deg.detach().cpu().numpy())
        d_lambda       = float(d_lam.detach().cpu().numpy())

    # --- target lambda from map + Δλ  ---
    base_lambda   = cal.get_target_AFR(rpm, fuel= fuel, lambda_table=lambda_target_map)/fuel.AFR_stoich
    target_lambda = float(base_lambda + d_lambda)

    # --- call combustion ---
    if analyze:
        combustion_Wiebe(
            spec=spec, fuel=fuel, rpm=rpm, throttle=throttle, ve=ve,
            soc_offset_deg=soc_offset_deg, fmep_offset_Pa=fmep_offset_Pa, target_lambda=target_lambda,                      
            plot=True, return_dic=False, **combustion_kwargs
        )
        return

    res = combustion_Wiebe(
        spec=spec, fuel=fuel, rpm=rpm, throttle=throttle, ve=ve,
        soc_offset_deg=soc_offset_deg, fmep_offset_Pa=fmep_offset_Pa, target_lambda=target_lambda,                          
        plot=False, return_dic=True, **combustion_kwargs
    )
    if not isinstance(res, dict):
        raise RuntimeError("combustion_Wiebe did not return a dict. Ensure return_dic=True is honored.")

    # per-cycle per-second (4-stroke)
    cps = rpm / 120.0
    mdot_air  = res["m_air_per_cycle_kg"]  * cps * spec.n_cylinder
    mdot_fuel = res["m_fuel_per_cycle_kg"] * cps * spec.n_cylinder

    # emissions & BSFC
    AFR = cal.get_target_AFR(rpm, fuel=fuel)  # returns a scalar AFR
    CO2_gps, CO_gps, NOx_gps, HC_gps = estimate_Emissions(mdot_fuel, AFR, 0.98)

    PkW = max(res["power_kw"], 1e-9)
    BSFC_g_per_kWh = (mdot_fuel * 3600.0 * 1000.0) / PkW
    to_gkWh = lambda gps: (gps * 3600.0) / PkW
    out = {
        "RPM": rpm,
        "Throttle": throttle,
        "VE": ve,
        # performance
        "Torque_Nm": res["torque_nm"],
        "Power_kW":  res["power_kw"],
        # mean effective pressures
        "IMEP_bar":  res["imep_gross_pa"] / 1e5,
        "BMEP_bar":  res["bmep_pa"]       / 1e5,
        "FMEP_bar":  res["fmep_pa"]       / 1e5,
        "PMEP_bar":  res["pmep_pa"]       / 1e5,
        # flows
        "mdot_air_kg_s":  mdot_air,
        "mdot_fuel_kg_s": mdot_fuel,
        # fuel economy
        "BSFC_g_per_kWh": BSFC_g_per_kWh,
        # emissions
        "CO2_gps": CO2_gps, "CO_gps": CO_gps, "NOx_gps": NOx_gps, "HC_gps": HC_gps,
        # emissions intensities
        "CO2_g_kWh": to_gkWh(CO2_gps), "CO_g_kWh": to_gkWh(CO_gps),
        "NOx_g_kWh": to_gkWh(NOx_gps), "HC_g_kWh": to_gkWh(HC_gps),
        # timings / phasing
        "ca10_deg": res["ca10_deg"], "ca50_deg": res["ca50_deg"], "ca90_deg": res["ca90_deg"],
        "Pmax_bar": res["pmax_bar"], "Tmax_K": res["tmax_k"],
        # mixture / combustion 
        "lambda":         res["lambda"],
        "phi":            res["phi"],
        "soc_deg":        res["soc_deg"],
        "eoc_deg":        res["eoc_deg"],
        "burn_10_90_deg": res["burn_10_90_deg"],
        # flame speeds
        "S_L_m_per_s": res["S_L_m_per_s"],
        "S_T_m_per_s": res["S_T_m_per_s"],
        # knock proxy
        "knock_index": res["knock_index"],
        # energy (per cycle, per cylinder)
        "q_chem_kj_per_cycle": res["q_chem_kj_per_cycle"],
        "q_ht_kj_per_cycle":   res["q_ht_kj_per_cycle"],
        "w_ind_kj_per_cycle":  res["w_ind_kj_per_cycle"],
        "eta_ind_percent":     res["eta_ind_percent"],
    }
    return out
def SingleRun(spec: EngineSpec, 
              fuel: FuelSpec, 
              rpm: int, 
              throttles: Optional[Iterable[float]] = None, 
              analyze_throttles: Optional[Iterable[float]] = None, 
              combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
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
            RunPoint(spec = spec,
                     fuel = fuel,
                     rpm = int(rpm),
                     throttle = float(th),
                     analyze = (th in analyze_set),
                     combustion_kwargs=combustion_kwargs)
        )
    return pd.DataFrame(rows).sort_values(["RPM","Throttle"]).reset_index(drop=True)
def WideOpenThrottle(spec:EngineSpec, 
                     fuel : FuelSpec,
                     RPM_min: int, 
                     RPM_max: int, 
                     step: int = 100, 
                     *, 
                     analyze_points: Optional[Iterable[int]] = None, 
                     combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
    """
    WOT sweep using VE from spec.ve_table. Returns one row per RPM.
    """
    analyze_set = set(analyze_points or [])
    rows = []
    for r in range(int(RPM_min), int(RPM_max)+1, int(step)):
        rows.append(RunPoint(spec = spec, 
                             fuel = fuel,
                             rpm = int(r), 
                             throttle=1.0, 
                             analyze=(r in analyze_set), combustion_kwargs=combustion_kwargs)
        )
    return pd.DataFrame(rows).sort_values("RPM").reset_index(drop=True)
def FullRangeSweep(spec:EngineSpec, 
                   fuel: FuelSpec, 
                   RPM_min: int, 
                   RPM_max: int, 
                   step: int = 100, 
                   throttles: Optional[Iterable[float]] = None, 
                   *, 
                   combustion_kwargs: Optional[dict] = None) -> pd.DataFrame:
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
                RunPoint(spec = spec,
                         fuel = fuel,
                         rpm = int(r),
                         throttle = float(th),
                         analyze = False,
                         combustion_kwargs=combustion_kwargs)
            )
    return pd.DataFrame(rows).sort_values(["RPM","Throttle"]).reset_index(drop=True)
'''
def DesignComparison():
    return
'''



