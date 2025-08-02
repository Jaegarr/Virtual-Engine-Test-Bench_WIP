import os
import datetime
import re

def clean_filename(name):
    """
    Clean a string to make it a valid filename by replacing invalid characters.

    Parameters:
        name (str): The original filename string.

    Returns:
        str: A cleaned filename string with invalid characters replaced by underscores.
    """
    # Replace invalid filename characters (\/:"*?<>|) with underscores
    return re.sub(r'[\\/:"*?<>|]+', "_", name)

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
    file = clean_filename(file) if file else "results"
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{file}_{now}.csv"

    print(f"📁 Enter folder path to save the file (leave empty to use default folder: '{default_folder}' inside the project):")
    folder = input().strip()
    if folder:
        folder = clean_filename(folder)
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