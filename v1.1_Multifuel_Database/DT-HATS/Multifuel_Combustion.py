import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- 1) Load data ----
xlsx_path = Path(r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\DT-HATS\Multifuel_Combustion.xlsx")

# If your data is on the first sheet, this is fine. Otherwise add sheet_name="YourSheet"
df = pd.read_excel(xlsx_path)

# ---- 2) Column hygiene ----
# Some exports can create a duplicated Lambda column. Prefer 'Lambda' if present, else 'Lambda_dup'
if 'Lambda' not in df.columns:
    if 'Lambda_dup' in df.columns:
        df['Lambda'] = df['Lambda_dup']
    else:
        raise KeyError(f"Couldn't find 'Lambda' column. Available columns: {list(df.columns)}")

# Make sure the necessary columns exist
required = ['Fuel', 'Lambda', 'IMEP (bar)', 'Laminar Flame Speed (m/s)']
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")

# Normalize labels for the legend
label_map = {
    'Hydrogen&Ammonia(10%w Hydrogen)': 'NH3 + 10% H2',
    'Hydrogen': 'Hydrogen',
    'Gasoline': 'Gasoline',
    'Ammonia': 'Ammonia'
}
df['FuelLabel'] = df['Fuel'].map(label_map).fillna(df['Fuel'])

# Ensure numeric sort
df['Lambda'] = pd.to_numeric(df['Lambda'], errors='coerce')

# ---- 3) Plot setup ----
fig, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax2 = ax1.twinx()

# consistent color per fuel
unique_fuels = list(dict.fromkeys(df['FuelLabel']))  # preserves order of first appearance
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {fuel: default_colors[i % len(default_colors)] for i, fuel in enumerate(unique_fuels)}

# Fixed-duration region shading and dashed threshold
lambda_split = 1.2
ax1.axvspan(lambda_split, float(df['Lambda'].max()), alpha=0.1)

marker_imep = 'o'   # circles for IMEP
marker_fs   = 's'   # squares for flame speed

def plot_segmented(ax, x, y, color, marker, label_base, split=lambda_split):
    mask_pre  = x <= split
    mask_post = x >= split
    if mask_pre.any():
        ax.plot(x[mask_pre], y[mask_pre], '-', marker=marker, color=color, label=label_base)
    if mask_post.any():
        ax.plot(x[mask_post], y[mask_post], '--', marker=marker, color=color)

# ---- 4) Draw series ----
for fuel in unique_fuels:
    sub = df[df['FuelLabel'] == fuel].sort_values('Lambda')
    x = sub['Lambda'].to_numpy()

    # Left axis: IMEP
    y_imep = sub['IMEP (bar)'].to_numpy()
    plot_segmented(ax1, x, y_imep, color_map[fuel], marker_imep, f'{fuel} – IMEP')

    # Right axis: Laminar Flame Speed
    y_fs = sub['Laminar Flame Speed (m/s)'].to_numpy()
    plot_segmented(ax2, x, y_fs, color_map[fuel], marker_fs, f'{fuel} – Flame speed')

# ---- 5) Labels, limits, legend ----
ax1.set_xlabel('Lambda (λ)')
ax1.set_ylabel('IMEP [bar]')
ax2.set_ylabel('Laminar flame speed [m/s]')
ax1.set_xlim(float(df['Lambda'].min()), float(df['Lambda'].max()))

# Build combined legend without duplicates
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
seen, lines, labs = set(), [], []
for L, T in list(zip(lines1, labs1)) + list(zip(lines2, labs2)):
    if T not in seen:
        lines.append(L); labs.append(T); seen.add(T)
ax1.legend(lines, labs, loc='upper right', fontsize=9, frameon=True)

ax1.set_title('IMEP & Laminar Flame Speed vs Lambda @ 3000 rpm / WOT\n(dashed: fixed burn duration region λ ≥ 1.2)')
plt.tight_layout()

out = xlsx_path.with_name('demo1_imep_flamespeed_vs_lambda.png')
plt.savefig(out, dpi=300)
# plt.show()  # uncomment if you want an interactive window
print(f"Saved: {out}")