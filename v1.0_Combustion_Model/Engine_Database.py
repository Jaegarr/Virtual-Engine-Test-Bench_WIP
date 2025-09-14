from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
@dataclass
class EngineSpec:
    n_cylinder : int
    bore_m : float
    stroke_m : float
    conrod_m : float
    compression_ratio : float
    ve_table: Optional[pd.DataFrame] = None
    def V_displacement_cylinder_m3(self) -> float:
        return np.pi * self.bore_m**2 / 4.0 * self.stroke_m
    def V_displacement_total_m3(self) -> float:
        return self.V_displacement_cylinder_m3 * self.n_cylinder
class EngineDB:
    def __init__(self):
        self.db: Dict[str, EngineSpec] = {}
    def register(self, name: str, spec: EngineSpec) -> None:
        self.db[name] = spec
    def get(self, name: str) -> EngineSpec:
        if name not in self.db:
            raise KeyError(f"Engine '{name}' not found. Available: {list(self.db.keys())}")
        return self.db[name]  # <-- IMPORTANT: return it!
    def list(self) -> List[str]:
        return sorted(self.db.keys())
Engines = EngineDB()
Engines.register("Nissan_VQ35DE__NA_3.5L_V6_350Z", EngineSpec( n_cylinder=6, bore_m=0.0955, stroke_m=0.0814, conrod_m=0.1442, compression_ratio=10.3,))