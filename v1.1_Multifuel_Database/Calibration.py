import pandas as pd
import numpy as np
from Fuel_Database import Fuels, FuelSpec
from scipy.interpolate import RegularGridInterpolator

lambda_target_map = pd.DataFrame({ 'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000], 'Lambda': [1, 1, 1, 1, 1, 1, 1]}) #0.97, 0.93, 0.91, 0.90, 0.88, 0.88, 0.88]  
def get_target_AFR(rpm: float,
                   fuel: FuelSpec,
                   lambda_table: dict = lambda_target_map) -> float:
    """
        Determine the target air - fuel ratio (AFR) for a given engine speed and fuel.
        Parameters
        ----------
        rpm : float
            Engine speed [rev/min].
        fuel : FuelSpec
            Fuel specification containing at least `AFR_stoich`.
        lambda_table : dict, optional
            Mapping of RPM to target lambda values (default: lambda_target_map).
        Returns
        -------
        float
            Target AFR = stoichiometric AFR of the fuel x target lambda.
        Notes
        -----
        - Target lambda is linearly interpolated from the given table.
        - λ < 1.0 at high rpm indicates enrichment for knock protection/cooling.
        - Because AFR_stoich is taken from the fuel spec, different fuels
        automatically yield correct targets.
    """
    target_lambda = float(np.interp(rpm, lambda_target_map['RPM'], lambda_target_map['Lambda']))
    target_AFR = fuel.AFR_stoich * target_lambda
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
'''
def SparkTiming():
    return
'''
