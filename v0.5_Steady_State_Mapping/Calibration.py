import pandas as pd
import numpy as np
import math, os
from scipy.interpolate import RegularGridInterpolator
lambda_target_map = pd.DataFrame({'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000],'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})
def get_target_lambda(rpm,AFR = 14.7):
   target_lambda = float(np.interp(rpm,lambda_target_map['RPM'],lambda_target_map['Lambda']))
   target_AFR = AFR*target_lambda
   return target_AFR
def load_ve_table(file_path=None):
    """
    Load VE table from CSV, return ve_vs_rpm Series.
    If file_path is None, load default file.
    """
    if not file_path:
        file_path = os.path.join('C:/Users/berke/OneDrive/Masaüstü/GitHub/Virtual-Engine-Test-Bench/v0.5_Steady_State_Mapping/Nissan_350Z_VE.csv')  # default
    ve_table = pd.read_csv(file_path, index_col=0)
    return ve_table
def get_ve_from_table(rpm, throttle, ve_table):
    ve_results = []
    map_values = ve_table.index.to_numpy(dtype=float)
    rpm_values = ve_table.columns.to_numpy(dtype=float)
    ve_values = ve_table.to_numpy()/100
    for t in throttle:
        map_kpa = 20 + t*(100-20) #20 kPa at idle, 100 kPa at WOT
        interpolator = RegularGridInterpolator((map_values,rpm_values),ve_values,bounds_error=False,fill_value=None)
        ve = interpolator([[map_kpa,rpm]])[0]
        ve_results.append(ve)
    return ve_results

    