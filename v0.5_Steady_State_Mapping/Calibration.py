import pandas as pd
import numpy as np
import math
lambda_target_map = pd.DataFrame({'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000],'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})
def get_target_lambda(rpm,AFR = 14.7):
   target_lambda = float(np.interp(rpm,lambda_target_map['RPM'],lambda_target_map['Lambda']))
   target_AFR = AFR*target_lambda
   return target_AFR
def get_ve_from_table(rpm,ve_vs_rpm):
   rpms = ve_vs_rpm.index.to_numpy(dtype=float)
   ves = ve_vs_rpm.values
   ve_interp = np.interp(rpm, rpms, ves)
   return ve_interp
def calculate_fmep(rpm,displacement_l):
   fmep = 0.25 + 0.02*rpm/1000 + 0.03*(rpm/1000)**2
   displacement_m3 = displacement_l/1e3
   fmep_pa = fmep*1e5
   fmep_nm = fmep_pa * displacement_m3/(4*math.pi)
   return fmep_nm
def calculate_pmep(rpm, displacement_l):
   pmep = 0.02 + 0.00001*rpm
   pmep_pa = pmep*1e5
   displacement_m3 = displacement_l/1e3
   pmep_nm = pmep_pa*displacement_m3/(4*math.pi)
   return pmep_nm
    