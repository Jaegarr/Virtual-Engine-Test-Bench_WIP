import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from engine_model import combustion_Wiebe         
from Engine_Database import Engines
from Fuel_Database import Fuels
from Calibration import get_ve_from_table

USE_VVL_FLAG   = True          # adds a binary feature: 1 if rpm >= 4500, else 0
EPOCHS  = 800
LR      = 5e-4
SPSA_C  = 5e-4
SPSA_K  = 3     # average 3 perturbations per step for less noise
PRINT_EVERY    = 100
DYNO_XLSX      = r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\Nissan_350z_Dyno.xlsx"
SAVE_DIR       = r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database"
SAVE_FILE      = "mlp_correction.pt"   # weights
SAVE_META      = "mlp_correction_meta.json"
ENGINE_KEY     = "Nissan_VQ35DE_NA_3.5L_V6_350Z"
FUEL_KEY       = "Gasoline"
lambda_target_map = pd.DataFrame({
    "RPM":    [1000, 2000, 3000, 4000, 5000, 6000, 7000],
    "Lambda": [0.97, 0.93, 0.91, 0.90, 0.88, 0.88, 0.88]
})
# =========================
# Small 3-head MLP
# =========================
class MLPCorrection(nn.Module):
    """
    Inputs: [rpm_norm] or [rpm_norm, vvl_flag]
    Outputs: [ΔFMEP_bar, ΔSOC_deg, Δλ]
    """
    def __init__(self, with_vvl_flag: bool = False, hidden: int = 8):
        super().__init__()
        in_features = 2 if with_vvl_flag else 1
        self.with_vvl_flag = with_vvl_flag
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)  # 2 -> 3 (add Δλ head)
        )
        # bias towards slightly positive ΔFMEP at low rpm
        with torch.no_grad():
            self.net[-1].bias.data[0] = 0.05  # head 0 = ΔFMEP

    def forward(self, rpm: float, vvl_active: float | None = None):
        # normalize RPM to ~[0,1]
        rpm_norm = (float(rpm) - 1000.0) / (7000.0 - 1000.0)
        rpm_norm = max(0.0, min(1.0, rpm_norm))

        if self.with_vvl_flag:
            if vvl_active is None:
                vvl_active = 1.0 if rpm >= 4500 else 0.0
            x = torch.tensor([[rpm_norm, float(vvl_active)]], dtype=torch.float32)
        else:
            x = torch.tensor([[rpm_norm]], dtype=torch.float32)

        raw_fmep, raw_soc, raw_dlambda = self.net(x)[0]

        # ΔFMEP: ≥0, increase with rpm, cap growth
        Fmep_max_add = 0.35 # bar
        slope = rpm_norm ** 1.2
        d_fmep_bar = F.softplus(raw_fmep) * slope
        d_fmep_bar = torch.clamp(d_fmep_bar, 0.0, Fmep_max_add)
        # ΔSOC bounded
        d_soc_deg = torch.clamp(raw_soc, -5.0, 5.0)
        # Δλ tight bounds via tanh
        d_lambda = 0.08 * torch.tanh(raw_dlambda)

        return d_fmep_bar, d_soc_deg, d_lambda
# Data
dyno_df = pd.read_excel(DYNO_XLSX)
dyno_df.columns = [c.strip() for c in dyno_df.columns]
rpm_col = [c for c in dyno_df.columns if c.lower().startswith("rpm")][0]
tq_col  = [c for c in dyno_df.columns if "torque" in c.lower()][0]
rpms        = dyno_df[rpm_col].to_numpy(dtype=float)
torque_dyno = dyno_df[tq_col].to_numpy(dtype=float)
order = np.argsort(rpms)
rpms = rpms[order]
torque_dyno = torque_dyno[order]
# Physics wrapper
def base_lambda_from_map(rpm: float) -> float:
    return float(np.interp(rpm,
        lambda_target_map["RPM"].to_numpy(dtype=float),
        lambda_target_map["Lambda"].to_numpy(dtype=float))
    )
def run_model_with_offsets(rpm: float,
                           fmep_off_Pa: float,
                           soc_off_deg: float,
                           d_lambda: float) -> float:
    """
    Calls combustion with:
      - FMEP offset (Pa)
      - SOC offset (deg)
      - target_lambda override = clip(base_lambda + Δλ, 0.85, 1.10)
    """
    ve = float(get_ve_from_table(rpm, throttle=1.0,
              ve_table=getattr(Engines.get(ENGINE_KEY), "ve_table", None)))

    base_lam = base_lambda_from_map(rpm)
    lam = float(np.clip(base_lam + float(d_lambda), 0.85, 1.10))

    res = combustion_Wiebe(
        spec=Engines.get(ENGINE_KEY),
        fuel=Fuels.get(FUEL_KEY),
        rpm=float(rpm),
        throttle=1.0,
        ve=ve,
        soc_offset_deg=float(soc_off_deg),
        fmep_offset_Pa=float(fmep_off_Pa),
        target_lambda=lam,            # <-- NEW
        plot=False,
        return_dic=True
    )
    return float(res["torque_nm"])
# Loss evaluation (black-box)
loss_fn = nn.MSELoss()
def eval_loss(model: MLPCorrection, rpms_np: np.ndarray, tq_np: np.ndarray, loss_fn) -> float:
    model.eval()
    pred_list   = []
    d_soc_list  = []
    d_fmep_list = []
    d_lam_list  = []  # NEW
    with torch.no_grad():
        for rpm in rpms_np:
            vvl_flag = 1.0 if (USE_VVL_FLAG and rpm >= 4500) else 0.0
            d_fmep_bar, d_soc_deg, d_lambda = model(rpm, vvl_flag) 
            fmep_off_Pa = float((d_fmep_bar * 1e5).cpu().numpy())  # bar -> Pa
            soc_off_deg = float(d_soc_deg.cpu().numpy())
            d_lambda_f  = float(d_lambda.cpu().numpy())
            tq_pred = run_model_with_offsets(rpm, fmep_off_Pa, soc_off_deg, d_lambda_f)
            pred_list.append(tq_pred)
            d_soc_list.append(soc_off_deg)
            d_fmep_list.append(float(d_fmep_bar.cpu().numpy()))
            d_lam_list.append(d_lambda_f)  # NEW
    pred   = torch.tensor(pred_list, dtype=torch.float32)
    target = torch.tensor(tq_np,    dtype=torch.float32)
    mse = loss_fn(pred, target)
    d_soc   = torch.tensor(d_soc_list,  dtype=torch.float32)
    d_fmep  = torch.tensor(d_fmep_list, dtype=torch.float32)
    d_lambda= torch.tensor(d_lam_list,  dtype=torch.float32)
    # Smoothness (squared total variation)
    soc_pen = torch.mean((d_soc[1:]    - d_soc[:-1])**2)     if len(d_soc)   > 1 else torch.tensor(0.0)
    lam_pen = torch.mean((d_lambda[1:] - d_lambda[:-1])**2)  if len(d_lambda)> 1 else torch.tensor(0.0)
    # FMEP monotonic non-decreasing vs RPM
    if len(d_fmep) > 1:
        df = d_fmep[1:] - d_fmep[:-1]
        mono_pen = torch.mean(torch.relu(-df))
    else:
        mono_pen = torch.tensor(0.0)
    # keep magnitudes small (L2)
    mag_pen = 1e-3 * (torch.mean(d_soc**2) + torch.mean(d_fmep**2) + torch.mean(d_lambda**2))  # include Δλ
    # weights 
    total = mse + 1e-3 * soc_pen + 1.5e-3 * mono_pen + 1e-3 * lam_pen + mag_pen
    return float(total.item())
# SPSA update (gradient-free)
def spsa_step(model: MLPCorrection, rpms_np: np.ndarray, tq_np: np.ndarray, loss_fn,
              c: float = SPSA_C, K: int = SPSA_K):
    """
    One SPSA gradient estimate (K perturbations averaged) and write grads into model.params
    After this, call optimizer.step() to apply the update.
    """
    theta = parameters_to_vector(model.parameters()).detach()
    g = torch.zeros_like(theta)
    for _ in range(K):
        # Rademacher ±1
        delta = torch.randint_like(theta, low=0, high=2, dtype=torch.int64).float()
        delta = 2 * delta - 1
        # normalize to avoid huge steps if param dim is large
        delta = delta / (torch.linalg.norm(delta) + 1e-8)
        # L(theta + c*delta)
        vector_to_parameters(theta + c * delta, model.parameters())
        Lp = eval_loss(model, rpms_np, tq_np, loss_fn)
        # L(theta - c*delta)
        vector_to_parameters(theta - c * delta, model.parameters())
        Lm = eval_loss(model, rpms_np, tq_np, loss_fn)
        g += ((Lp - Lm) / (2 * c)) * delta
    g /= float(K)
    # Restore params to theta and set .grad tensors
    vector_to_parameters(theta, model.parameters())
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = g[idx:idx + n].view_as(p).clone()
        idx += n
# Train
def main():
    model = MLPCorrection(with_vvl_flag=USE_VVL_FLAG, hidden=8)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        spsa_step(model, rpms, torque_dyno, loss_fn, c=SPSA_C, K=SPSA_K)
        optimizer.step()

        if (epoch + 1) % PRINT_EVERY == 0:
            cur_loss = eval_loss(model, rpms, torque_dyno, loss_fn)
            print(f"[Epoch {epoch+1:4d}] Eval loss: {cur_loss:.4f}")
    # save
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_FILE)
    torch.save(model.state_dict(), save_path)
    meta = {
        "use_vvl_flag": USE_VVL_FLAG,
        "epochs": EPOCHS,
        "lr": LR,
        "spsa_c": SPSA_C,
        "spsa_k": SPSA_K,
        "engine_key": ENGINE_KEY,
        "fuel_key": FUEL_KEY,
        "rpm_min": float(rpms.min()),
        "rpm_max": float(rpms.max())
    }
    with open(os.path.join(SAVE_DIR, SAVE_META), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Trained correction saved to: {save_path}")
    print(f"📝 Metadata saved to: {os.path.join(SAVE_DIR, SAVE_META)}")

if __name__ == "__main__":
    main()