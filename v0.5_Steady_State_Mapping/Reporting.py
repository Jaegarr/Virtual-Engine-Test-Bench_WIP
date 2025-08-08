import os
import datetime
import re
import matplotlib.pyplot
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
    print("📄 Enter a filename to save your results (without .csv):")
    file = input().strip()
    file = re.sub(r'[\\/:"*?<>|]+', "_", file) if file else "results" # Replace invalid filename characters (\/:"*?<>|) with underscores
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file}_{now}.csv"
    print(f"📁 Enter folder path to save the file (leave empty to use default folder: '{default_folder}' inside the project):")
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
def rpm_vs_plots(df):
    fig, ax1 = matplotlib.pyplot.subplots()
    ax1.set_xlabel('Engine Speed (RPM)')
    ax1.set_ylabel('Torque (Nm)')
    ax1.plot(df['Engine Speed (RPM)'], df['Torque (Nm)'], color='tab:blue', label='Torque')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Power (kW)')
    ax2.plot(df['Engine Speed (RPM)'], df['Power (kW)'], color='tab:red', label='Power')
    matplotlib.pyplot.title('Torque & Power Curves - WOT')
    fig.tight_layout()
    matplotlib.pyplot.show()
def emission_plots(df):
    fig = matplotlib.pyplot.subplot()
    fig.set_xlabel('Engine Speed (RPM)')
    fig.set_ylabel('Emissions (g/s)')
    fig.plot(df['Engine Speed (RPM)'], df['CO2(g/s)'], label='CO2')
    fig.plot(df['Engine Speed (RPM)'], df['CO(g/s)'], color='tab:red', label='CO')
    fig.plot(df['Engine Speed (RPM)'], df['NOx(g/s)'], color='tab:blue', label='NOx')
    fig.plot(df['Engine Speed (RPM)'], df['HC(g/s)'], color='tab:green', label='HC')
    fig.legend()
    matplotlib.pyplot.title('Emissions')
    matplotlib.pyplot.show()