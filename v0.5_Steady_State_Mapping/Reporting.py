import os, datetime, re
def clean_filename(name):
    # Replace invalid characters with underscore
    return re.sub(r'[\\/:"*?<>|]+', "_", name)
def export_results_to_csv(data, default_folder="Results"):
    """
    Export DataFrame to CSV.
    By default saves into a 'Results' folder inside the project directory.
    Adds timestamp to filename to prevent overwriting.
    """
    print("📄 Enter a filename to save your results (without .csv):")
    file = input().strip()
    file = clean_filename(file) if file else "results"
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file}_{now}.csv"
    print(f"📁 Enter folder path to save the file (leave empty to use default folder: '{default_folder}' inside the project):")
    folder = input().strip()
    if folder:
        folder = clean_filename(folder)
        full_folder = folder
    else:
        # Automatically use 'Results' inside the current script/project folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_folder = os.path.join(script_dir, default_folder)
    # Make sure the folder exists
    os.makedirs(full_folder, exist_ok=True)
    full_path = os.path.join(full_folder, filename)
    data.to_csv(full_path, index=False)
    print(f"✅ Results exported to: {full_path}")