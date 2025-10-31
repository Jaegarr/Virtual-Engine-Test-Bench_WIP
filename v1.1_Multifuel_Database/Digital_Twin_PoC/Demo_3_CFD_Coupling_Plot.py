import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

baseline_path = Path(r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\Digital_Twin_PoC\Demo_3_Hydrogen_Baseline_2025-10-27_01-19-14.csv")       # <-- path to baseline dataset
cfd_path      = Path(r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\Digital_Twin_PoC\Demo_3_Hydrogen_CFD_Coupled_2025-10-31_17-25-44.csv")
base = pd.read_csv(baseline_path)
cfd  = pd.read_csv(cfd_path)
base.columns = base.columns.str.strip()
cfd.columns  = cfd.columns.str.strip()

rpm_base = base['Engine Speed (RPM)']
rpm_cfd  = cfd['Engine Speed (RPM)']
torque_base = base['Torque (Nm)']
torque_cfd  = cfd['Torque (Nm)']
st_base = base['S_T (m/s)']
st_cfd  = cfd['S_T (m/s)']
clip_limit = 60.0

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(rpm_base, torque_base, 'r-', lw=2, label='Torque (Baseline)')
ax1.plot(rpm_cfd, torque_cfd, 'b-', lw=2, label='Torque (CFD)')
ax1.set_xlabel('Engine Speed (RPM)')
ax1.set_ylabel('Torque (Nm)')
ax1.tick_params(axis='y')
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(rpm_base, st_base, 'r--', lw=1.8, label='Turbulent Flame Speed (Baseline)')
ax2.plot(rpm_cfd, st_cfd, 'b--', lw=1.8, label='Turbulent Flame Speed (CFD)')
ax2.set_ylabel('Turbulent Flame Speed (m/s)')
ax2.tick_params(axis='y')
ax2.fill_between(rpm_base, clip_limit, max(st_base.max(), st_cfd.max()), 
                 color='gray', alpha=0.1)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower center')

fig.suptitle('Turbulent Flame Speed Clipping Limit: 60 m/s', fontsize=11)
fig.tight_layout()
plt.show()