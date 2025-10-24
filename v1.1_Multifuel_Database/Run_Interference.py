from engine_model import combustion_Wiebe
from ML_Correction import MLPCorrection, apply_corrections
import torch

USE_ML_CORRECTION = True   # toggle

# Load trained model if used
if USE_ML_CORRECTION:
    ml_model = MLPCorrection(with_vvl_flag=True)
    ml_model.load_state_dict(torch.load("data/trained_correction.pth"))
    ml_model.eval()
else:
    ml_model = None

def run_point(spec, fuel, rpm, throttle, ve):
    soc_offset_deg = 0.0
    fmep_offset_Pa = 0.0
    if USE_ML_CORRECTION:
        fmep_offset_Pa, soc_offset_deg = apply_corrections(ml_model, rpm)
    return combustion_Wiebe(
        spec=spec,
        rpm=rpm,
        throttle=throttle,
        ve=ve,
        fuel=fuel,
        soc_offset_deg=soc_offset_deg,
        fmep_offset_Pa=fmep_offset_Pa,
        plot=False,
        return_dic=True
    )