import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
def load_ve_table(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(str(source), index_col=0)
    # headers must be numeric: rows=MAP[kPa], cols=RPM
    df.index   = pd.to_numeric(df.index, errors='raise')
    df.columns = pd.to_numeric(df.columns, errors='raise')
    # sort for monotonic grids
    df = df.sort_index().sort_index(axis=1)
    return df
@dataclass
class EngineSpec:
    n_cylinder : int
    bore_m : float
    stroke_m : float
    conrod_m : float
    compression_ratio : float
    ve_table: Optional[Union[pd.DataFrame, str, os.PathLike]] = None  # path or DF
    def __post_init__(self):
        if isinstance(self.ve_table, (str, os.PathLike)) or self.ve_table is not None and not isinstance(self.ve_table, pd.DataFrame):
            self.ve_table = load_ve_table(self.ve_table)
    def V_displacement_cylinder_m3(self) -> float:
        import numpy as np
        return np.pi * self.bore_m**2 / 4.0 * self.stroke_m
    def V_displacement_total_m3(self) -> float:
        return self.V_displacement_cylinder_m3() * self.n_cylinder
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
Engines.register("Nissan_VQ35DE_NA_3.5L_V6_350Z", EngineSpec(n_cylinder = 6, bore_m = 0.0955, stroke_m = 0.0814, conrod_m = 0.1442, compression_ratio = 10.3,ve_table=r'C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.0_Combustion_Model\Nissan_350Z_VE.csv'))
