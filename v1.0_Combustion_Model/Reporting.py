import os, datetime, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def export_results_to_csv(data, default_folder="Results"):
    """
    Export a pandas DataFrame to a CSV file with a user-defined filename and folder.
    Adds a timestamp suffix to the filename to avoid overwriting existing files.
    Creates the target folder if it doesn't exist.

    Parameters:
        data (pandas.DataFrame): The data to export.
        default_folder (str): Default folder name to save files in if no folder path is specified.

    Process:
        - Prompts the user to enter a filename (without extension).
        - Cleans the filename by replacing invalid characters.
        - Adds a timestamp suffix (YYYY-MM-DD_HH-MM-SS).
        - Prompts user for folder path or uses the default folder inside the script directory.
        - Ensures the folder exists (creates if necessary).
        - Saves the DataFrame to CSV without index.
        - Prints the full path of the saved file.

    Returns:
        None
    """
    print("\n📄 Enter a filename to save your results (without .csv):")
    file = input().strip()
    file = re.sub(r'[\\/:"*?<>|]+', "_", file) if file else "results" # Replace invalid filename characters (\/:"*?<>|) with underscores
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file}_{now}.csv"
    print(f"\n📁 Enter folder path to save the file (leave empty to use default folder: '{default_folder}' inside the project):")
    folder = input().strip()
    if folder:
        folder = re.sub(r'[\\/:"*?<>|]+', "_", folder)
        full_folder = folder
    else:
        # Use 'Results' folder inside the current script directory by default
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_folder = os.path.join(script_dir, default_folder)
    # Create the folder if it doesn't exist
    os.makedirs(full_folder, exist_ok=True)
    full_path = os.path.join(full_folder, filename)
    data.to_csv(full_path, index=False)
    print(f"✅ Results exported to: {full_path}")
def to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Map RunPoint/Mode outputs to legacy Reporting column names."""
    return pd.DataFrame({
    'Engine Speed (RPM)': df['RPM'],
    'Throttle':           df['Throttle'],
    'Torque (Nm)':        df['Torque_Nm'],
    'Power (kW)':         df['Power_kW'],
    'Horsepower':         df['Power_kW'] * 1.34102209,   # kW -> HP
    # mass flows
    'Air Flow(g/s)':      df['mdot_air_kg_s']  * 1000.0,
    'Fuel Flow(g/s)':     df['mdot_fuel_kg_s'] * 1000.0,
    # pressures/efficiencies
    'IMEP (bar)':         df['IMEP_bar'],
    'BMEP (bar)':         df['BMEP_bar'],
    'PMEP (bar)':         df['PMEP_bar'],
    'FMEP (bar)':         df['FMEP_bar'],
    'BSFC (g/kWh)':       df['BSFC_g_per_kWh'],
    # emissions (instantaneous rates)
    'CO2(g/s)':           df['CO2_gps'],
    'CO(g/s)':            df['CO_gps'],
    'NOx(g/s)':           df['NOx_gps'],
    'HC(g/s)':            df['HC_gps'],
    # emissions intensities
    'CO2 (g/kWh)':        df['CO2_g_kWh'],
    'CO (g/kWh)':         df['CO_g_kWh'],
    'NOx (g/kWh)':        df['NOx_g_kWh'],
    'HC (g/kWh)':         df['HC_g_kWh'],
}).round(3)
def rpm_vs_plots(df):
    unique_throttle = np.unique(df['Throttle'])
    if (len(unique_throttle) == 1):  # WOT case
        fig, ax1 = plt.subplots()
        plt.grid(True)
        ax1.set_ylabel('Power (kW)')
        ax1.plot(df['Engine Speed (RPM)'], df['Power (kW)'], label = 'Power', color = 'red')
        ax1.set_ylim(0, 240)
        ax1.set_yticks(np.arange(0, 240, 20))
        ax2 = ax1.twinx()
        ax2.set_xlabel('Engine Speed (RPM)')
        ax2.set_ylabel('Torque (Nm)')
        ax2.plot(df['Engine Speed (RPM)'], df['Torque (Nm)'], label = 'Torque', color = 'blue')
        ax2.set_ylim(0, 600)
        ax2.set_yticks(np.arange(0,600,50))
        ax1.set_title('Torque & Power Curves — WOT')
        fig.tight_layout()
        plt.show()
        return
    else:
        # Multi-throttle case — collect inputs first
        requested = []
        while True:
            request = input("Select at least 2 and up to 10 throttle positions (e.g., 0.4, 0.7, 1.0) or 'q' to finish: ").strip()
            if request.lower() == 'q':
                break
            try:
                new_vals = [float(x) for x in request.replace('%', '').split(',') if x.strip() != '']
            except ValueError:
                print("❌ Could not parse throttle inputs. Use numbers like: 0.4, 0.7, 1.0")
                continue
            requested.extend(new_vals)
            if len(requested) >= 2:
                more = input("Do you want to add more throttle values? (y/n): ").strip().lower()
                if more == 'n':
                    break
        if not (2 <= len(requested) <= 10):
            raise ValueError("Please provide between 2 and 10 throttle values.")
        # Clip to [0,1], dedupe, sort
        requested = np.clip(np.array(requested, dtype=float), 0.0, 1.0)
        requested = sorted(set(requested))
        # Torque figure
        plt.figure()
        plt.grid(True)
        for throttle in requested:
            sub = df[np.isclose(df['Throttle'].astype(float), throttle, atol=1e-3)].copy()
            if sub.empty:
                continue
            sub.sort_values('Engine Speed (RPM)', inplace=True)
            plt.plot(sub['Engine Speed (RPM)'], sub['Torque (Nm)'], label=f'Throttle {throttle:.2f}')
        plt.title('Torque vs RPM — Full Sweep')
        plt.xlabel('Engine Speed (RPM)')
        plt.ylabel('Torque (Nm)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Power figure
        plt.figure()
        plt.grid(True)
        for throttle in requested:
            sub = df[np.isclose(df['Throttle'].astype(float), throttle, atol=1e-3)].copy()
            if sub.empty:
                continue
            sub.sort_values('Engine Speed (RPM)', inplace=True)
            plt.plot(sub['Engine Speed (RPM)'], sub['Power (kW)'], label=f'Throttle {throttle:.2f}')
        plt.title('Power vs RPM — Full Sweep')
        plt.xlabel('Engine Speed (RPM)')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.tight_layout()
        plt.show()
def emission_plots(df):
    fig = plt.subplot()
    
    fig.set_xlabel('Engine Speed (RPM)')
    fig.set_ylabel('Emissions (g/s)')
    fig.plot(df['Engine Speed (RPM)'], df['CO2(g/s)'], color='magenta', label='CO2')
    fig.plot(df['Engine Speed (RPM)'], df['CO(g/s)'], color='tab:red', label='CO')
    fig.plot(df['Engine Speed (RPM)'], df['NOx(g/s)'], color='tab:blue', label='NOx')
    fig.plot(df['Engine Speed (RPM)'], df['HC(g/s)'], color='tab:green', label='HC')
    fig.legend()
    plt.title('Emissions')
    plt.yscale("log")
    plt.show()