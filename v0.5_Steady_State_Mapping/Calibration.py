import pandas as pd
import numpy as np
lambda_target_map = pd.DataFrame({'RPM': [1000, 2000, 3000, 4000, 5000, 6000, 7000],'Lambda': [1.0, 1.0, 0.95, 0.92, 0.90, 0.88, 0.88]})
def get_target_lambda(rpm,AFR = 14.7):
   target_lambda = float(np.interp(rpm,lambda_target_map['RPM'],lambda_target_map['Lambda']))
   target_AFR = AFR*target_lambda
   return target_AFR
def calculate_fmep(rpm):
   fmep = 0.25 + 0.02*rpm/1000 + 0.03*(rpm/1000)**2
   fmep_pa = fmep*1e5
   return fmep_pa

    