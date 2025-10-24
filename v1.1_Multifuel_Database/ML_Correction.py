import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_corrections(ml_model, rpm):
    vvl_active = 1.0 if rpm >= 4500 else 0.0
    return ml_model(rpm, vvl_active)   # returns (ΔFMEP_bar, ΔSOC_deg) as tensors
class MLPCorrection(nn.Module):
    """
    Inputs: [rpm_norm] or [rpm_norm, vvl_flag]
    Outputs: [ΔFMEP_bar, ΔSOC_deg]
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
        # bias towards slightly positive ΔFMEP
        self.net[-1].bias.data[0] = 0.05  

    def forward(self, rpm: float, vvl_active: float | None = None):
        # Normalize RPM
        rpm_norm = (float(rpm) - 1000.0) / (7000.0 - 1000.0)
        rpm_norm = max(0.0, min(1.0, rpm_norm))  # clamp just in case

        if self.with_vvl_flag:
            if vvl_active is None:
                vvl_active = 1.0 if rpm >= 4500 else 0.0
            x = torch.tensor([[rpm_norm, float(vvl_active)]], dtype=torch.float32)
        else:
            x = torch.tensor([[rpm_norm]], dtype=torch.float32)

        out = self.net(x)
        raw_fmep, raw_soc = out[0]

        FMEP_MAX_ADD_BAR = 0.35      # max EXTRA bar we let ML add
        slope = rpm_norm**1.2        # soft monotonic growth with rpm (0→1)
        d_fmep_bar = torch.softplus(raw_fmep) * slope
        d_fmep_bar = torch.clamp(d_fmep_bar, 0.0, FMEP_MAX_ADD_BAR)


        # SOC bounded ±5°
        d_soc_deg = torch.clamp(raw_soc, -5.0, 5.0)

        return d_fmep_bar, d_soc_deg