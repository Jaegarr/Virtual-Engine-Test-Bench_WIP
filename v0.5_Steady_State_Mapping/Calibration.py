import pandas as pd
import numpy as np
import math, os
from scipy.interpolate import RegularGridInterpolator

# Simplified target lambda map by RPM (could be replaced by a CSV/config)
lambda_target_map = pd.DataFrame({ 'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000], 'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})
def get_target_lambda(rpm, AFR=14.7):
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
def load_ve_table(file_path=None):
    '''
    Load a Volumetric Efficiency (VE) table from a CSV file.

    Parameters:
        file_path (str, optional): Path to the VE CSV file.
            - If None, loads the default Nissan 350Z VE table from the project folder.

    Returns:
        pandas.DataFrame: VE table with manifold pressure (kPa) as index and RPM as columns.

    Notes:
        - VE values are typically in percent in the CSV (e.g., 80 means 0.80).
        - CSV must have manifold pressure values as the first column header.
    '''
    if not file_path:
        file_path = os.path.join('C:/Users/berke/OneDrive/Masaüstü/GitHub/Virtual-Engine-Test-Bench/v0.5_Steady_State_Mapping/Nissan_350Z_VE.csv'      )
    ve_table = pd.read_csv(file_path, index_col=0)
    return ve_table
def get_ve_from_table(rpm, throttle, ve_table):
    '''
    Interpolate VE (Volumetric Efficiency) values from a VE table 
    for multiple throttle positions at a given RPM.

    Parameters:
        rpm (float): Engine speed in revolutions per minute.
        throttle (list of float): List of throttle positions (0.0-1.0).
        ve_table (pandas.DataFrame): VE table with manifold pressure as index and RPM as columns.

    Returns:
        list of float: Interpolated VE values (as decimal, e.g., 0.85) for each throttle position.

    Notes:
        - Converts throttle to manifold absolute pressure using a simple map (idle=20 kPa, WOT=100 kPa).
        - Uses RegularGridInterpolator to interpolate VE for given (MAP, RPM).
        - Returns VE values divided by 100 since CSV usually stores percentages.
    '''
    ve_results = []
    map_values = ve_table.index.to_numpy(dtype=float)      # MAP breakpoints (kPa)
    rpm_values = ve_table.columns.to_numpy(dtype=float)    # RPM breakpoints
    ve_values = ve_table.to_numpy() / 100                  # VE as decimal
    interpolator = RegularGridInterpolator((map_values, rpm_values), ve_values, bounds_error=False, fill_value=None)
    if isinstance(throttle, (list, tuple, np.ndarray)):
        ve_list = []
        for t in throttle:
            map_kpa = 20 + t * (100 - 20)
            ve = interpolator([[map_kpa, rpm]])[0]
            ve_list.append(ve)
        return ve_list
    else:
        # single value
        map_kpa = 20 + throttle * (100 - 20)
        ve = interpolator([[map_kpa, rpm]])[0]
        return ve