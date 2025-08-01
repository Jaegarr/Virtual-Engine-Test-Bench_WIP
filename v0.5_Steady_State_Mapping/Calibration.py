import pandas as pd
import numpy as np
import math
lambda_target_map = pd.DataFrame({'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000],'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})
def get_target_lambda(rpm,AFR = 14.7):
   target_lambda = float(np.interp(rpm,lambda_target_map['RPM'],lambda_target_map['Lambda']))
   target_AFR = AFR*target_lambda
   return target_AFR
def load_ve_table(file_path=None):
    """Load VE table from CSV, return VE vs RPM series"""
    if not file_path:
        file_path = 'Data/Nissan_350Z_VE.csv'  # default relative path
        try:
            ve_table = pd.read_csv(file_path, index_col=0)  # load table
            ve_mode = 'table'
            print(f'✅ Loaded VE table from {file_path}')
            break
        except Exception as e:
            print(f'❌ Could not load file: {e}')
    ve_table = pd.read_csv(file_path, index_col=0)
    # Select MAP=100 kPa row (you might adjust to nearest if missing)
    ve_vs_rpm = ve_table.loc[100] / 100  # convert % to fraction
    return ve_vs_rpm
def get_ve_from_table(rpm,ve_vs_rpm):
   rpms = ve_vs_rpm.index.to_numpy(dtype=float)
   ves = ve_vs_rpm.values
   ve_interp = np.interp(rpm, rpms, ves)
   return ve_interp

    