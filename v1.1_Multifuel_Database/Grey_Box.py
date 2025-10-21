# gp_residual.py
import os, joblib, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, DotProduct
import pandas as pd
from Test_Modes import RunPoint
from Engine_Database import Engines
from Fuel_Database import Fuels

def _scale_rpm(rpm):
    r = np.asarray(rpm, dtype=np.float64).reshape(-1,1)
    rmin, rmax = float(r.min()), float(r.max())
    span = max(1.0, rmax - rmin)
    return (r - rmin)/span*2.0 - 1.0, {"rmin": rmin, "rspan": span}
def _apply_scale(rpm, scaler):
    r = np.asarray(rpm, dtype=np.float64).reshape(-1,1)
    return (r - scaler["rmin"])/max(1.0, scaler["rspan"])*2.0 - 1.0
def fit_gp_residual(rpm, torque_phys, torque_dyno, save_path="gp_residual.pkl"):
    rpm = np.asarray(rpm).astype(np.float64)
    resid = (np.asarray(torque_dyno) - np.asarray(torque_phys)).astype(np.float64)
    Xs, scaler = _scale_rpm(rpm)
    kernel = C(10.0, (1e-3, 1e3)) * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 5.0)) \
             + WhiteKernel(noise_level=2.0, noise_level_bounds=(1e-6, 50.0)) \
             + 0.5 * DotProduct(sigma_0=1.0)

    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, n_restarts_optimizer=5, random_state=0)
    gp.fit(Xs, resid)
    joblib.dump({"gp": gp, "scaler": scaler}, save_path)
    return save_path
class GPResidualCorrector:
    def __init__(self, model_path="gp_residual.pkl"):
        if os.path.exists(model_path):
            obj = joblib.load(model_path)
            self.gp = obj["gp"]; self.scaler = obj["scaler"]
        else:
            self.gp = None; self.scaler = None
    def correct(self, rpm, torque_phys):
        if self.gp is None: 
            return float(torque_phys)
        X = _apply_scale([rpm], self.scaler)
        mu, sigma = self.gp.predict(X, return_std=True)
        mu, sigma = float(mu[0]), float(sigma[0])
        sigma_ref = 5.0 
        w = 1.0 / (1.0 + (sigma / sigma_ref)**2)
        delta = mu * w
        lim = 0.15 * abs(torque_phys) + 20.0
        delta = float(np.clip(delta, -lim, +lim))
        return float(torque_phys + delta)
    
spec = Engines.get("Nissan_VQ35DE_NA_3.5L_V6_350Z")
fuel = Fuels.get("Gasoline")
dyno = pd.read_excel("C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\Nissan_350z_Dyno.xlsx")   # must have RPM, TorqueNm_dyno
rpms = dyno["RPM"].values
T_phys = []
for r in rpms:
    res = RunPoint(spec=spec, fuel=fuel, rpm=int(r), throttle=1.0)
    T_phys.append(res["Torque_Nm"])

fit_gp_residual(rpms, T_phys, dyno["TorqueNm_dyno"].values, save_path="gp_residual.pkl")
print("GP residual model saved.")