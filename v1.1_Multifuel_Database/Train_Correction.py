"""
train_correction.py
-------------------
Trains a 2-head MLP (ΔFMEP, ΔSOC) using dyno torque data vs model predictions.
Uses SPSA (gradient-free) so the physics model can stay a black box.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from engine_model import combustion_Wiebe         
from Engine_Database import Engines
from Fuel_Database import Fuels
from Calibration import get_ve_from_table

USE_VVL_FLAG   = True          # adds a binary feature: 1 if rpm >= 4500, else 0
EPOCHS  = 1500
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

# =========================
# Small 2-head MLP
# =========================
class MLPCorrection(nn.Module):
    """
    Inputs: [rpm_norm] or [rpm_norm, vvl_flag]
    Outputs: [ΔFMEP_Pa, ΔSOC_deg]
    """
    def __init__(self, with_vvl_flag: bool = False, hidden: int = 8):
        super().__init__()
        in_features = 2 if with_vvl_flag else 1
        self.with_vvl_flag = with_vvl_flag
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, rpm: float, vvl_active: float | None = None):
        # normalize RPM to ~[0,1]
        rpm_norm = (float(rpm) - 1000.0) / (7000.0 - 1000.0)
        if self.with_vvl_flag:
            x = torch.tensor([[rpm_norm, 1.0 if (vvl_active is None and rpm >= 4500) else float(vvl_active)]],
                             dtype=torch.float32)
        else:
            x = torch.tensor([[rpm_norm]], dtype=torch.float32)
        out = self.net(x)              # shape (1,2)
        d_fmep_Pa, d_soc_deg = out[0] # tensors
        # Clamp SOC to ±5 deg as a soft safety bound (hard clamp; loss also encourages small values)
        d_soc_deg = torch.clamp(d_soc_deg, -5.0, 5.0)
        return d_fmep_Pa, d_soc_deg   # keep as tensors (no .item() here)

# =========================
# Data
# =========================
dyno_df = pd.read_excel(DYNO_XLSX)
# be robust to any stray spaces/casing in column names
dyno_df.columns = [c.strip() for c in dyno_df.columns]
# Expect columns like "RPM" and "Torque (Nm)" (might have leading space in your file)
rpm_col = [c for c in dyno_df.columns if c.lower().startswith("rpm")][0]
tq_col  = [c for c in dyno_df.columns if "torque" in c.lower()][0]

rpms        = dyno_df[rpm_col].to_numpy(dtype=float)
torque_dyno = dyno_df[tq_col].to_numpy(dtype=float)

# ensure increasing order (helps monotonic reg)
order = np.argsort(rpms)
rpms = rpms[order]
torque_dyno = torque_dyno[order]

# =========================
# Physics wrapper
# =========================
def run_model_with_offsets(rpm: float, fmep_off_Pa: float, soc_off_deg: float) -> float:
    """
    Calls your combustion model with the two offsets injected.
    Returns model torque [Nm] for the given rpm (WOT assumed).
    """
    ve = float(get_ve_from_table(rpm,throttle=1.0,ve_table = getattr(Engines.get(ENGINE_KEY), "ve_table", None)))
    res = combustion_Wiebe(
        spec=Engines.get(ENGINE_KEY),
        fuel=Fuels.get(FUEL_KEY),
        rpm=float(rpm),
        throttle=1.0,
        ve=ve,                     # adjust if you have a VE map
        soc_offset_deg=float(soc_off_deg),
        fmep_offset_Pa=float(fmep_off_Pa),
        plot=False,
        return_dic=True
    )
    # your combustion_Wiebe returns lowercase keys in return_dic
    return float(res["torque_nm"])

# =========================
# Loss evaluation (black-box)
# =========================
loss_fn = nn.MSELoss()

def eval_loss(model: MLPCorrection, rpms_np: np.ndarray, tq_np: np.ndarray, loss_fn) -> float:
    """
    Runs the physics model across all rpms, compares torque to dyno.
    Adds small penalties:
      - SOC smoothness (squared TV across RPMs)
      - FMEP monotonic (penalize negative steps with ReLU)
      - magnitude penalty to keep offsets small
    Returns scalar float loss.
    """
    model.eval()

    pred_list = []
    d_soc_list = []
    d_fmep_list = []

    with torch.no_grad():
        for rpm in rpms_np:
            vvl_flag = 1.0 if (USE_VVL_FLAG and rpm >= 4500) else 0.0
            d_fmep_bar, d_soc_deg = model(rpm, vvl_flag)

            # convert for physics call
            fmep_off_Pa = float((d_fmep_bar * 1e5).cpu().numpy())  # bar -> Pa
            soc_off_deg = float(d_soc_deg.cpu().numpy())

            tq_pred = run_model_with_offsets(rpm, fmep_off_Pa, soc_off_deg)

            pred_list.append(tq_pred)
            d_soc_list.append(soc_off_deg)
            d_fmep_list.append(float(d_fmep_bar.cpu().numpy()))

    pred   = torch.tensor(pred_list, dtype=torch.float32)
    target = torch.tensor(tq_np,    dtype=torch.float32)
    mse = loss_fn(pred, target)

    # soft constraints
    d_soc  = torch.tensor(d_soc_list,  dtype=torch.float32)
    d_fmep = torch.tensor(d_fmep_list, dtype=torch.float32)

    # SOC smoothness (squared total variation)
    soc_pen = torch.mean((d_soc[1:] - d_soc[:-1])**2) if len(d_soc) > 1 else torch.tensor(0.0)

    # FMEP monotonic non-decreasing vs RPM (penalize negative finite differences)
    if len(d_fmep) > 1:
        df = d_fmep[1:] - d_fmep[:-1]
        mono_pen = torch.mean(torch.relu(-df))
    else:
        mono_pen = torch.tensor(0.0)

    # keep magnitudes small (L2)
    mag_pen = 0.001 * (torch.mean(d_soc**2) + torch.mean(d_fmep**2))
    total = mse + 1e-3 * soc_pen + 3e-3 * mono_pen + mag_pen
    return float(total.item())

# =========================
# SPSA update (gradient-free)
# =========================
from torch.nn.utils import parameters_to_vector, vector_to_parameters

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

# =========================
# Train
# =========================
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