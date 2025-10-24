"""
train_correction.py
-------------------
Trains a 2-head MLP (ΔFMEP, ΔSOC) using dyno torque data vs model predictions.
Saves model weights for later inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from ML_Correction import MLPCorrection
from engine_model import combustion_Wiebe
from Engine_Database import Engines
from Fuel_Database import Fuels  

USE_VVL_FLAG = True
EPOCHS = 800
LR = 1e-3
SAVE_PATH = r'C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database'
PRINT_EVERY = 100

dyno_df = pd.read_excel(r'C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\Nissan_350z_Dyno.xlsx')
rpms = dyno_df["RPM"].to_numpy()
torque_dyno = dyno_df[" Torque (Nm)"].to_numpy()

model = MLPCorrection(with_vvl_flag=USE_VVL_FLAG)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ===============================================================
# Helper: run combustion model with ML corrections
# ===============================================================
def run_model_with_offsets(rpm, fmep_off_Pa, soc_off_deg):
    throttle = 1.0
    ve = 1.0  # or from your VE table if available
    res = combustion_Wiebe(
        spec=Engines.get("Nissan_VQ35DE_NA_3.5L_V6_350Z"),
        fuel=Fuels.get("Gasoline"),
        rpm=rpm,
        throttle=throttle,
        ve=ve,
        soc_offset_deg=soc_off_deg,
        fmep_offset_Pa=fmep_off_Pa,
        plot=False,
        return_dic=True
    )
    return res["torque_nm"]

# ===============================================================
# TRAINING LOOP
# ===============================================================
for epoch in range(EPOCHS):
    total_loss = 0.0

    for i, rpm in enumerate(rpms):
        vvl_flag = 1.0 if rpm >= 4500 else 0.0
        # Predict offsets from MLP
        fmep_off_Pa, soc_off_deg = model(rpm, vvl_flag)

        # --- Detach for physics model (since combustion_Wiebe is non-differentiable)
        fmep_val = float(fmep_off_Pa.detach().cpu().numpy())
        soc_val = float(soc_off_deg.detach().cpu().numpy())

        # --- Run physics model
        torque_pred = run_model_with_offsets(rpm, fmep_val, soc_val)

        # --- Build pseudo-loss (compare prediction vs target)
        # treat torque_pred as target-like scalar, not requiring grad
        pred = torch.tensor([torque_pred], dtype=torch.float32)
        target = torch.tensor([torque_dyno[i]], dtype=torch.float32)

        loss_torque = loss_fn(pred, target)
        # Regularization (smoothness, bounds, physics)
        # Encourage small offsets and smooth ΔFMEP (positive, increasing)
        loss_phys = 0.0
        if fmep_off_Pa < 0:
            loss_phys += abs(fmep_off_Pa) * 1e-4
        if abs(soc_off_deg) > 5.0:
            loss_phys += (abs(soc_off_deg) - 5.0) * 1e-3
        loss = loss_torque + loss_phys
        total_loss += loss

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f"[Epoch {epoch+1:4d}] Loss: {total_loss.item():.4f}")
# SAVE MODEL
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Trained correction saved to {SAVE_PATH}")