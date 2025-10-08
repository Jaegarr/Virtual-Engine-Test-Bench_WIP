import sys
import re
import pandas as pd
from Engine_Database import Engines, EngineSpec
from Test_Modes import FullRangeSweep, WideOpenThrottle, SingleRun, RunPoint
from Reporting import export_results_to_csv,rpm_vs_plots, emission_plots, to_legacy
pd.set_option('display.float_format', '{:.3f}'.format)
# INPUT CHECK
def _input_float(prompt, lowerlimit=None, upperlimit=None):
    while True:
        try:
            value = float(input(prompt))
            if lowerlimit is not None and value < lowerlimit: print(f"Value must be >= {lowerlimit}."); continue
            if upperlimit is not None and value > upperlimit: print(f"Value must be <= {upperlimit}."); continue
            return value
        except:
            print("Invalid number. Try again.")
def _input_integer(prompt, lowerlimit=None, uppperlimit=None):
    while True:
        try:
            value = int(input(prompt))
            if lowerlimit is not None and value < lowerlimit: print(f"Value must be >= {lowerlimit}."); continue
            if uppperlimit is not None and value > uppperlimit: print(f"Value must be <= {uppperlimit}."); continue
            return value
        except:
            print("Invalid integer. Try again.")
#%% ENGINE SELECTION
while True:
    print("Choose engine:")
    print("1. Use preloaded engine")
    print("2. Load your own engine geometry")
    print("3. Exit")
    selection = input("Enter 1, 2 or 3: ").strip()
    if selection == '1':
        names = Engines.list()
        print("\nAvailable engines:")
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
        index = _input_integer("Select engine by number: ", 1, len(names))
        engine_name = names[index-1]
        spec = Engines.get(engine_name)
        break
    elif selection == '2':
        print("\nEnter custom engine geometry:")
        n_cyl   = _input_integer("Number of cylinders [1..16]: ", 1, 16)
        bore_mm = _input_float("Bore [mm]: ", 50, 120)
        stroke_mm = _input_float("Stroke [mm]: ", 40, 150)
        rod_mm  = _input_float("Conrod length [mm]: ", 80, 300)
        cr      = _input_float("Compression ratio [-]: ", 7.0, 15.0)
        path = input("Enter VE table CSV path for this engine: ").strip()
        try:
            ve_tbl = pd.read_csv(path, index_col=0)
        except Exception as e:
            print(f"Failed to load VE table: {e}")
            continue
        spec = EngineSpec(n_cylinder=n_cyl, bore_m=bore_mm/1000.0, stroke_m=stroke_mm/1000.0, conrod_m=rod_mm/1000.0, compression_ratio=cr, ve_table=ve_tbl)
        print("Enter a name to save your engine :")
        enginename = input().strip()
        enginename = re.sub(r'[\s\\/:\"*?<>|]+', '_', enginename)
        enginename = re.sub(r'[^A-Za-z0-9._-]', '', enginename)
        enginename = re.sub(r'_+', '_', enginename).strip('._')
        enginename = (enginename or 'Engine')[:80]
        Engines.register(enginename, spec)
        break
    elif selection == '3':
        print("Exiting.")
        sys.exit(0)
    else:
        print("Invalid choice.")
#%% TEST SELECTION
# SPEC CHECK
if 'spec' not in globals() or getattr(spec, 've_table', None) is None:
    print("❌ Engine spec or VE table missing. Build 'spec' and attach spec.ve_table before running tests.")
    sys.exit(1)
while True:
    print('\nPlease select the test you want to execute:')
    print("1 - Single run (one RPM, sweep throttle)")
    print("2 - Wide Open Throttle (1.0 throttle, sweep RPM)")
    print("3 - Full sweep (RPM x Throttle grid)")
    print("4 - Exit")
    testMode = input("Enter your test choice (1, 2, 3, 4): ").strip()
    if testMode == '1':
        print("\nYou selected: Single run")
        while True:
            try:
                rpm = int(input('Enter the RPM: '))
                if rpm < 800 or rpm > 15000:
                    print('RPM must be between 800 and 15000.')
                    continue
                break
            except:
                print('Invalid input. RPM must be an integer.')
        df_singleRun = SingleRun(spec=spec, rpm=rpm) 
        df = to_legacy(df_singleRun)
        export_results_to_csv(df)
        rpm_vs_plots(df)
        ans = input('Would you like to plot emissions (Yes/No): ').strip().lower()
        if ans == 'yes':
            emission_plots(df)
        if combustion_analysis == 'yes':
            RPMpoint = float(input('\nChoose RPM point to analyze:'))
            Throttlepoint = float(input('\nChoose throttle point to analyze:'))
            RunPoint(spec=spec, rpm = RPMpoint, throttle= Throttlepoint, analyze = True)
        sys.exit()

    elif testMode == '2':
        print("\nYou selected: Wide Open Throttle")
        while True:
            try:
                rpmMin = int(input('Enter minimum RPM: '))
                if rpmMin < 800:
                    print('Minimum RPM must be at least 800.')
                    continue
                break
            except:
                print('Invalid input. Enter an integer RPM.')
        while True:
            try:
                rpmMax = int(input('Enter maximum RPM: '))
                if rpmMax > 15000:
                    print('Maximum RPM cannot exceed 15000.')
                    continue
                if rpmMax <= rpmMin:
                    print('Maximum RPM must be greater than minimum RPM.')
                    continue
                break
            except:
                print('Invalid input. Enter an integer RPM.')
        df_wot = WideOpenThrottle(spec=spec, RPM_min=rpmMin, RPM_max=rpmMax, step=100)
        df = to_legacy(df_wot)
        export_results_to_csv(df)
        rpm_vs_plots(df)
        ans = input('\nWould you like to plot emissions (Yes/No): ').strip().lower()
        if ans == 'yes':
            emission_plots(df)
        combustion_analysis = input('\nWould you like to analyze the combustion (Yes/No): ').strip().lower()
        if combustion_analysis == 'yes':
            RPMpoint = float(input('\nChoose RPM point to analyze(throttle = 1):'))
            RunPoint(spec=spec, rpm = RPMpoint, throttle= 1, analyze = True)
        sys.exit()
    elif testMode == '3':
        print("You selected: Full sweep")
        while True:
            try:
                rpmMin = int(input('Enter minimum RPM: '))
                if rpmMin < 800:
                    print('Minimum RPM must be at least 800.')
                    continue
                break
            except:
                print('Invalid input. Enter an integer RPM.')
        while True:
            try:
                rpmMax = int(input('Enter maximum RPM: '))
                if rpmMax > 15000:
                    print('Maximum RPM cannot exceed 15000.')
                    continue
                if rpmMax <= rpmMin:
                    print('Maximum RPM must be greater than minimum RPM.')
                    continue
                break
            except:
                print('Invalid input. Enter an integer RPM.')
        df_full = FullRangeSweep(spec=spec, RPM_min=rpmMin, RPM_max=rpmMax, step=100)
        df = to_legacy(df_full)
        export_results_to_csv(df)
        rpm_vs_plots(df)   # works with multiple throttles too
        combustion_analysis = input('\nWould you like to analyze the combustion (Yes/No): ').strip().lower()
        if combustion_analysis == 'yes':
            RPMpoint = float(input('\nChoose RPM point to analyze:'))
            Throttlepoint = float(input('\nChoose throttle point to analyze:'))
            RunPoint(spec=spec, rpm = RPMpoint, throttle= Throttlepoint, analyze = True)
        sys.exit()
    elif testMode == '4':
        print("Exiting program.")
        sys.exit()

    else:
        print("Invalid input. Please enter 1, 2, 3, or 4.")