import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# Simplified target lambda map by RPM (could be replaced by a CSV/config)
lambda_target_map = pd.DataFrame({ 'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000], 'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})  #1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88
def get_target_AFR(rpm, AFR=14.7):
    '''
    Calculate the target air–fuel ratio (AFR) based on engine speed.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        AFR (float): Stoichiometric AFR for the fuel (default is 14.7 for gasoline).

    Returns:
        float: Target AFR accounting for lambda enrichment.

    Notes:
        - Uses linear interpolation over a predefined lambda vs. RPM table.
        - At higher RPM, lambda < 1.0 means richer mixture for knock protection and cooling.
    '''
    target_lambda = float(np.interp(rpm, lambda_target_map['RPM'], lambda_target_map['Lambda']))
    target_AFR = AFR * target_lambda
    return target_AFR
def get_ve_from_table(rpm, throttle, ve_table, idle_kpa: float = 20.0, wot_kpa: float = 100.0):
    """
    Interpolate VE from a VE table (rows=MAP [kPa], cols=RPM).
    - ve_table can be a pandas.DataFrame or a CSV path.
    - throttle can be a scalar in [0..1] or an iterable of such.
    - VE in the table may be % (e.g., 95) or fraction (0.95); auto-detected.
    Returns float if throttle is scalar, else list[float].
    """
    # 1) Load/normalize table
    if isinstance(ve_table, pd.DataFrame):
        df = ve_table.copy()
    else:
        df = pd.read_csv(str(ve_table), index_col=0)
    # enforce numeric axes (MAP kPa x RPM) and sort
    df.index   = pd.to_numeric(df.index, errors='raise')
    df.columns = pd.to_numeric(df.columns, errors='raise')
    df = df.sort_index().sort_index(axis=1)
    map_values = df.index.to_numpy(dtype=float)   # kPa
    rpm_values = df.columns.to_numpy(dtype=float) # RPM
    ve_values  = df.to_numpy(dtype=float)
    # 2) Auto-detect % vs fraction
    if np.nanmax(ve_values) > 1.5:  # likely percentages (e.g., 95)
        ve_values = ve_values / 100.0
    # 3) Interpolator (allow mild extrapolation)
    interp = RegularGridInterpolator((map_values, rpm_values), ve_values,bounds_error=False, fill_value=None)
    # 4) Helpers
    def throttle_to_map(t):
        return idle_kpa + float(t) * (wot_kpa - idle_kpa)
    def interp_one(t):
        mk = throttle_to_map(t)
        val = interp([[mk, float(rpm)]])[0]
        if np.isnan(val):  # clip if we fell outside both axes
            mkc = np.clip(mk, map_values.min(), map_values.max())
            rpc = np.clip(float(rpm), rpm_values.min(), rpm_values.max())
            val = interp([[mkc, rpc]])[0]
        return float(val)
    # 5) Vector-friendly return
    if isinstance(throttle, (list, tuple, np.ndarray)):
        return [interp_one(t) for t in throttle]
    else:
        return interp_one(throttle)
def SparkTiming():
    return