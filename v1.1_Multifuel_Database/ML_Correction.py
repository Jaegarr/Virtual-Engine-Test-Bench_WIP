import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Initial lambda map
lambda_target_map = pd.DataFrame({
    "RPM":    [1000, 2000, 3000, 4000, 5000, 6000, 7000],
    "Lambda": [0.97, 0.93, 0.91, 0.90, 0.88, 0.88, 0.88]
})
class MLPCorrection(nn.Module):
    """
    Inputs:  rpm_norm (and optional vvl flag)
    Outputs: ΔFMEP_bar, ΔSOC_deg, Δλ
    """
    def __init__(self, with_vvl_flag: bool = False, hidden: int = 8):
        super().__init__()
        in_features = 2 if with_vvl_flag else 1
        self.with_vvl_flag = with_vvl_flag
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 3)  # <-- 2 -> 3
        )
        # bias towards slightly positive ΔFMEP at low rpm
        with torch.no_grad():
            self.net[-1].bias.data[0] = 0.05  # ΔFMEP head bias

    def forward(self, rpm: float, vvl_active: float | None = None):
        rpm_norm = (float(rpm) - 1000.0) / (7000.0 - 1000.0)
        rpm_norm = max(0.0, min(1.0, rpm_norm))
        if self.with_vvl_flag:
            if vvl_active is None:
                vvl_active = 1.0 if rpm >= 4500 else 0.0
            x = torch.tensor([[rpm_norm, float(vvl_active)]], dtype=torch.float32)
        else:
            x = torch.tensor([[rpm_norm]], dtype=torch.float32)

        raw_fmep, raw_soc, raw_dlambda = self.net(x)[0]
        # ΔFMEP ≥ 0, softly grows with RPM, and upper-bounded
        Fmep_max_add = 0.35 # bar
        slope = rpm_norm ** 1.2
        d_fmep_bar = F.softplus(raw_fmep) * slope
        d_fmep_bar = torch.clamp(d_fmep_bar, 0.0, Fmep_max_add)
        # ΔSOC bounded
        d_soc_deg = torch.clamp(raw_soc, -5.0, 5.0)
        # Δλ bounded (tight): ±0.08 
        d_lambda = 0.08 * torch.tanh(raw_dlambda)

        return d_fmep_bar, d_soc_deg, d_lambda