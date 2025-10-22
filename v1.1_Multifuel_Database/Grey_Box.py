from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Dict, Optional
import Calibration as cal
from engine_model import combustion_Wiebe
from Engine_Database import EngineSpec
from Fuel_Database import FuelSpec

# ---------------------------
# Utilities: piecewise-linear
# ---------------------------

@dataclass
class PLin:
    """Piecewise-linear function y(x) defined on fixed knots."""
    knots_x: np.ndarray  # shape (K,)
    values_y: np.ndarray # shape (K,)

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        return np.interp(x, self.knots_x, self.values_y)

    @staticmethod
    def from_theta(knots_x: Sequence[float], coeffs: Sequence[float]) -> "PLin":
        return PLin(np.asarray(knots_x, float), np.asarray(coeffs, float))

    def as_dict(self) -> Dict[str, list]:
        return {"knots_rpm": self.knots_x.tolist(), "values": self.values_y.tolist()}


# ---------------------------
# Packing / bounds / penalty
# ---------------------------

@dataclass
class KnobParam:
    """Holds the 3 knob curves and their bounds."""
    knots_rpm: np.ndarray            # shared knot vector
    theta_soc: np.ndarray            # deg
    theta_dur: np.ndarray            # scale
    theta_fmep: np.ndarray           # Pa
    # bounds for each coefficient (per-knot)
    lb_soc: float = -8.0
    ub_soc: float = +6.0
    lb_dur: float = 0.70
    ub_dur: float = 1.30
    lb_fmep: float = -0.2e5
    ub_fmep: float = +0.2e5
    def pack(self) -> np.ndarray:
        return np.concatenate([self.theta_soc, self.theta_dur, self.theta_fmep])
    @classmethod
    def unpack(cls, knots_rpm: np.ndarray, theta_vec: np.ndarray) -> "KnobParam":
        K = len(knots_rpm)
        soc = theta_vec[0:K]
        dur = theta_vec[K:2*K]
        fm  = theta_vec[2*K:3*K]
        return KnobParam(knots_rpm, soc, dur, fm)
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        K = len(self.knots_rpm)
        lb = np.concatenate([
            np.full(K, self.lb_soc),
            np.full(K, self.lb_dur),
            np.full(K, self.lb_fmep),
        ])
        ub = np.concatenate([
            np.full(K, self.ub_soc),
            np.full(K, self.ub_dur),
            np.full(K, self.ub_fmep),
        ])
        return lb, ub
    def curves(self) -> Tuple[PLin, PLin, PLin]:
        return (
            PLin(self.knots_rpm, self.theta_soc),
            PLin(self.knots_rpm, self.theta_dur),
            PLin(self.knots_rpm, self.theta_fmep),
        )
    def as_json(self) -> Dict[str, dict]:
        soc_pl, dur_pl, fm_pl = self.curves()
        return {"soc_shift_deg": soc_pl.as_dict(),
                "duration_scale": dur_pl.as_dict(),
                "fmep_offset_pa": fm_pl.as_dict()}
def second_difference_smoothness(theta: np.ndarray, knots: np.ndarray) -> float:
    """
    Smoothness penalty: sum of squared second differences (per set, per uniformized RPM).
    Works even if RPM spacing is uneven by normalizing to index space.
    """
    K = len(knots)
    if K < 3:
        return 0.0
    # split into three blocks
    soc, dur, fm = np.split(theta, 3)
    def p(z):
        return np.sum((z[:-2] - 2*z[1:-1] + z[2:])**2)
    return p(soc) + p(dur) + p(fm)
# Simulation wrapper (WOT)
def _ve_from_table(spec: EngineSpec, rpm: float, throttle: float = 1.0) -> float:
    tbl = getattr(spec, "ve_table", None)
    if tbl is None:
        raise ValueError("EngineSpec.ve_table is missing")
    return float(cal.get_ve_from_table(rpm, throttle, tbl))
def simulate_torque_wot(
    spec: EngineSpec,
    fuel: FuelSpec,
    rpm: float,
    soc_shift_deg: float,
    duration_scale: float,
    fmep_offset_pa: float,
) -> float:
    ve = _ve_from_table(spec, rpm, throttle=1.0)
    res = combustion_Wiebe(
        spec=spec,
        rpm=rpm,
        throttle=1.0,
        ve=ve,
        fuel=fuel,
        soc_shift_deg=float(soc_shift_deg),
        duration_scale=float(np.clip(duration_scale, 0.70, 1.30)),
        fmep_offset_pa=float(np.clip(fmep_offset_pa, -0.2e5, 0.2e5)),
        plot=False,
        return_dic=True
    )
    return float(res["torque_nm"])
def run_wot_with_knobs(
    spec: EngineSpec,
    fuel: FuelSpec,
    rpms: np.ndarray,
    knobs: KnobParam
) -> np.ndarray:
    f_soc, f_dur, f_fm = knobs.curves()
    torques = []
    for r in rpms:
        T = simulate_torque_wot(
            spec, fuel, float(r),
            soc_shift_deg=float(f_soc(r)),
            duration_scale=float(f_dur(r)),
            fmep_offset_pa=float(f_fm(r)),
        )
        torques.append(T)
    return np.asarray(torques, float)
# Fitting (optimization)
@dataclass
class FitConfig:
    # RPM knots for piecewise-linear curves
    knob_knots_rpm: np.ndarray = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000], dtype=float)
    # Smoothness regularization weight
    lambda_smooth: float = 1e-2
    # Optimizer options
    maxiter: int = 200
    tol: float = 1e-6
    # Initial guesses
    init_soc_deg: float = 0.0
    init_dur_scale: float = 1.0
    init_fmep_pa: float = 0.0
def fit_wot_knobs(
    spec: EngineSpec,
    fuel: FuelSpec,
    rpm_data: Sequence[float],
    torque_dyno: Sequence[float],
    cfg: FitConfig = FitConfig(),
) -> Tuple[KnobParam, Dict[str, float]]:
    """
    Calibrate ΔSOC(rpm), duration_scale(rpm), FMEP_offset(rpm) to match dyno torque at WOT.
    Returns (knobs, metrics).
    """
    rpms = np.asarray(rpm_data, float)
    Td = np.asarray(torque_dyno, float)
    knots = np.asarray(cfg.knob_knots_rpm, float)
    # Initialize parameters (constant curves)
    K = len(knots)
    kp0 = KnobParam(
        knots_rpm=knots,
        theta_soc=np.full(K, cfg.init_soc_deg, float),
        theta_dur=np.full(K, cfg.init_dur_scale, float),
        theta_fmep=np.full(K, cfg.init_fmep_pa, float),
    )
    theta0 = kp0.pack()
    lb, ub = kp0.bounds()
    # Loss function
    def obj(theta: np.ndarray) -> float:
        knobs = KnobParam.unpack(knots, theta)
        Tsim = run_wot_with_knobs(spec, fuel, rpms, knobs)
        err = Tsim - Td
        mse = float(np.mean(err**2))
        reg = cfg.lambda_smooth * second_difference_smoothness(theta, knots)
        return mse + reg
    # Optimize
    res = opt.minimize(
        obj, theta0, method="L-BFGS-B",
        bounds=list(zip(lb, ub)),
        options=dict(maxiter=cfg.maxiter, ftol=cfg.tol)
    )
    theta_star = res.x
    knobs_star = KnobParam.unpack(knots, theta_star)
    # Metrics
    Tsim_star = run_wot_with_knobs(spec, fuel, rpms, knobs_star)
    err = Tsim_star - Td
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(1.0, np.abs(Td)))) * 100.0
    metrics = {
        "rmse_Nm": rmse,
        "mape_pct": mape,
        "success": bool(res.success),
        "nit": int(res.nit),
        "final_loss": float(res.fun),
    }
    return knobs_star, metrics
# Save / Load knobs
def save_knobs_json(knobs: KnobParam, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(knobs.as_json(), f, indent=2)
def load_knobs_json(path: str) -> Tuple[PLin, PLin, PLin]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    def to_pl(d: Dict[str, list]) -> PLin:
        return PLin(np.asarray(d["knots_rpm"], float), np.asarray(d["values"], float))
    return to_pl(data["soc_shift_deg"]), to_pl(data["duration_scale"]), to_pl(data["fmep_offset_pa"])
# Runtime hook (optional)
class GreyBoxRuntime:
    """Cache of loaded knob curves for fast lookup at runtime."""
    def __init__(self, soc_pl: PLin, dur_pl: PLin, fm_pl: PLin):
        self.soc = soc_pl
        self.dur = dur_pl
        self.fm  = fm_pl

    @classmethod
    def from_json(cls, path: str) -> "GreyBoxRuntime":
        return cls(*load_knobs_json(path))

    def at_rpm(self, rpm: float) -> Tuple[float, float, float]:
        return float(self.soc(rpm)), float(self.dur(rpm)), float(self.fm(rpm))