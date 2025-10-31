import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
xlsx_path = Path(r"C:\Users\berke\OneDrive\Masaüstü\GitHub\Virtual-Engine-Test-Bench\v1.1_Multifuel_Database\DT-HATS\Multifuel_Combustion.xlsx")
df = pd.read_excel(xlsx_path)
if 'Lambda' not in df.columns and 'Lambda_dup' in df.columns:
    df['Lambda'] = df['Lambda_dup']
df['Lambda'] = pd.to_numeric(df['Lambda'], errors='coerce')
label_map = {
    'Hydrogen&Ammonia(10%w Hydrogen)': 'NH3 + 10% H2',
    'Hydrogen': 'Hydrogen',
    'Gasoline': 'Gasoline',
    'Ammonia': 'Ammonia'
}
df['FuelLabel'] = df['Fuel'].map(label_map).fillna(df['Fuel'])
fuels = ['Hydrogen', 'NH3 + 10% H2', 'Gasoline', 'Ammonia']
marker_map = {'Hydrogen': 'o', 'NH3 + 10% H2': '^', 'Gasoline': 's', 'Ammonia': 'D'}
color_map = {
    'Hydrogen': '#1f77b4',     # blue
    'NH3 + 10% H2': '#2ca02c', # green
    'Gasoline': '#9467bd',     # purple
    'Ammonia': '#e377c2'       # pink
}

fig, ax = plt.subplots(figsize=(8, 5))
lambda_split = 1.3
ax.axvspan(lambda_split, float(df['Lambda'].max()), color='0.8', alpha=0.15)
for fuel in fuels:
    sub = df[df['FuelLabel'] == fuel].sort_values('Lambda')
    if sub.empty:
        continue
    x = sub['Lambda'].to_numpy()
    y = sub['IMEP (bar)'].to_numpy()
    color = color_map[fuel]
    marker = marker_map[fuel]
    pre = x <= lambda_split
    post = x >= lambda_split
    if pre.any():
        ax.plot(x[pre], y[pre], '-', marker=marker, color=color, label=fuel, lw=2.2)
    if post.any():
        ax.plot(x[post], y[post], '--', marker=marker, color=color, lw=2.2)
ax.set_xlabel('Lambda (λ)')
ax.set_ylabel('IMEP [bar]')
ax.grid(True, which='both', lw=0.5, alpha=0.3)
ax.legend(title='Fuel', loc='best', frameon=False)
plt.tight_layout()
out_path = xlsx_path.with_name('demo1_IMEP_vs_lambda_fixedcolors.png')
plt.savefig(out_path, dpi=300)
print(f"Saved: {out_path}")
