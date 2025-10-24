import torch
import torch.nn as nn
import numpy as np

def apply_corrections(ml_model, rpm):
    vvl_active = 1.0 if rpm >= 4500 else 0.0
    return ml_model(rpm, vvl_active)
#  2-head MLP model 
class MLPCorrection(nn.Module):
    def __init__(self, with_vvl_flag=False):
        super().__init__()
        in_features = 2 if with_vvl_flag else 1
        hidden = 8
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)  # [ΔFMEP_bar, ΔSOC_deg]
        )
    def forward(self, rpm, vvl_active=None):
        rpm_norm = (rpm - 1000.0) / (7000.0 - 1000.0)
        if vvl_active is None:
            x = torch.tensor([[rpm_norm]], dtype=torch.float32)
        else:
            x = torch.tensor([[rpm_norm, float(vvl_active)]], dtype=torch.float32)
        out = self.net(x)
        delta_fmep_bar, delta_soc_deg = out[0]
        delta_fmep_Pa = delta_fmep_bar * 1e5  # convert bar→Pa
        delta_soc_deg = torch.clamp(delta_soc_deg, -5.0, 5.0)
        return delta_fmep_Pa, delta_soc_deg

