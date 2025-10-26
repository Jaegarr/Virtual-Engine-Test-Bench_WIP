import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
file_path = Path(r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\DT-HATS\Validation.xlsx")
df = pd.read_excel(file_path)
for col in ['Baseline Torque (Nm)', 'ML Corrected Torque (Nm)', 'Baseline Power (kW)', 'ML Corrected Power (kW)']:
    df[col] = df[col].astype(str).str.replace('%', '').astype(float)

# --- Plot setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, height_ratios=[2.2, 1])

# =====================
# Top: Absolute Torque
# =====================
ax1.plot(df['RPM'], df['Baseline Torque (Nm)'], color='#1f77b4', lw=2.0, label='Baseline')
ax1.plot(df['RPM'], df['ML Corrected Torque (Nm)'], color='#2ca02c', lw=2.0, label='ML Corrected')
ax1.plot(df['RPM'], df['Dyno Torque (Nm)'], 'k-', lw=2.5, alpha=0.8, label='Dyno')

ax1.set_ylabel('Torque [Nm]')
ax1.set_title('Model Evolution vs Dyno Reference')
ax1.grid(True, which='both', lw=0.4, alpha=0.3)
ax1.legend(frameon=False, loc='upper left')

# =====================
# Bottom: Relative Error
# =====================
ax2.plot(df['RPM'], df['Baseline Torque (Nm)'], '--', color='#1f77b4', lw=2.0, label='Baseline Deviation')
ax2.plot(df['RPM'], df['ML Corrected Torque (Nm)'], '-', color='#2ca02c', lw=2.0, label='ML Corrected Deviation')

ax2.axhline(0, color='k', lw=0.8)
ax2.fill_between(df['RPM'], -2, 2, color='gray', alpha=0.15, label='±2% zone')

ax2.set_xlabel('Engine Speed [rpm]')
ax2.set_ylabel('Torque Deviation [%]')
ax2.set_xlim(df['RPM'].min(), df['RPM'].max())
ax2.set_ylim(-3, 6)
ax2.grid(True, which='both', lw=0.4, alpha=0.3)
ax2.legend(frameon=False, loc='upper left')

plt.tight_layout()

# --- Save ---
out_path = file_path.with_name('demo2_torque_model_evolution.png')
plt.savefig(out_path, dpi=300)
# plt.show()
print(f"Saved: {out_path}")