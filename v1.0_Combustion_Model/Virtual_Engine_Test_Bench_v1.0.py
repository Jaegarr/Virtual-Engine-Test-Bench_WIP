import sys
import re
import pandas as pd
import Calibration as cal
from Test_Modes import FullRangeSweep, WideOpenThrottle, SingleRun
from Reporting import export_results_to_csv,rpm_vs_plots, emission_plots
from Engine_Database import Engines, EngineDB, EngineSpec
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
# ENGINE SELECTION
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
        # If this preloaded engine has no VE table attached, ask for one
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
#%% Test Selection
while True:
    print('Please select the test you want to execute:')
    print("1 - Single run")
    print("2 - Wide Open Throttle")
    print("3 - Full sweep")
    print("4 - Exit")
    testMode = input("Enter your test choice(1, 2, 3, 4): ")
    if testMode == '1':
        print("You have selected Single run")
        while True:
            try:
                print('Enter the RPM:')
                rpm = int(input())
                if rpm < 800 or rpm > 15000: 
                    print('RPM value must be between 800 RPM and 15000 RPM. Please enter the RPM:')
                    continue
                break
            except:
                print('RPM value must be between 800 RPM and 15000 RPM.')
        if ve_mode == 'table':
            results = SingleRun(rpm, displacement, ve_mode, ve_table=ve_vs_rpm)
        else:
            results = SingleRun(rpm, displacement, ve_mode, constant_ve=ve)
            df = pd.DataFrame(results, columns=['Engine Speed (RPM)', 'Throttle', 'Torque (Nm)', 'Power (kW)', 'Horsepower', 'Air Flow(g/s)', 'Fuel Flow(g/s)', 'CO2(g/s)', 'CO(g/s)', 'NOx(g/s)', 'HC(g/s)'])
            export_results_to_csv(df)
            sys.exit()
    elif testMode == '2':
        print("You selected Wide Open Throttle")
        try:
            spec = Engines.get("Nissan_VQ35DE__NA_3.5L_V6_350Z")
        except Exception as e:
            print(f"Engine spec not found: {e}")
            sys.exit(1)
        # RPM range input
        while True:
            try:
                print('Enter minimum RPM:')
                rpmMin = int(input())
                if rpmMin < 800:
                    print('Minimum RPM must be at least 800 RPM. Please enter minimum RPM:')
                    continue
                break
            except:
                print('Invalid input. Minimum RPM must be at least 800 RPM.')
        while True:
            try:
                print('Enter maximum RPM:')
                rpmMax = int(input())
                if rpmMax > 15000:
                    print('Maximum RPM cannot be higher than 15000 RPM. Please enter maximum RPM:')
                    continue
                if rpmMax <= rpmMin:
                    print('Maximum RPM must be greater than minimum RPM.')
                    continue
                break
            except:
                print('Invalid input. Maximum RPM cannot be higher than 15000 RPM.')
        df_wot = WideOpenThrottle(spec=spec, RPM_min=rpmMin, RPM_max=rpmMax, step=100, ve_mode=ve_mode, constant_ve=ve if ve_mode == 'constant' else 0.98, ve_table=ve_vs_rpm if ve_mode == 'table' else None, analyze_points=[], combustion_kwargs=None)
        df = pd.DataFrame({
            'Engine Speed (RPM)': df_wot['RPM'],
            'Throttle':           df_wot['Throttle'],
            'Torque (Nm)':        df_wot['Torque_Nm'],
            'Power (kW)':         df_wot['Power_kW'],
            'Horsepower':         df_wot['Power_kW'] * 1.34102209,     # kW -> HP
            'Air Flow(g/s)':      df_wot['mdot_air_kg_s']  * 1000.0,   # kg/s -> g/s
            'Fuel Flow(g/s)':     df_wot['mdot_fuel_kg_s'] * 1000.0,   # kg/s -> g/s
            'CO2(g/s)':           df_wot['CO2_gps'],
            'CO(g/s)':            df_wot['CO_gps'],
            'NOx(g/s)':           df_wot['NOx_gps'],
            'HC(g/s)':            df_wot['HC_gps'],
        })
        export_results_to_csv(df)
        rpm_vs_plots(df)
        emissionplot = input('Would you like to plot emissions(Yes/No):')
        if emissionplot.lower() == 'yes':
            emission_plots(df)
            sys.exit()
        elif emissionplot.lower() == 'no':
            sys.exit()
        else:
            print('Please type Yes or No:')
            elif emissionplot.lower() == 'no':
                sys.exit()
        else:
            print('Please type Yes or No:')
            continue
    elif testMode == '3':
        print("You selected Full sweep")
        while True:
            try:
                print('Enter minimum RPM:')
                rpmMin = int(input())
                if rpmMin < 800: 
                    print('Minimum RPM must be at least 800 RPM. Please enter minimum RPM:')
                    continue
                break
            except:
                print('Invalid input. Minimum RPM must be at least 800 RPM.')
        while True:
            try:
                print('Enter maximum RPM:')
                rpmMax = int(input())
                if rpmMax > 15000: 
                    print('Maximum RPM cannot be higher than 15000 RPM. Please enter maximum RPM:')
                    continue
                break
            except:
                print('Invalid input. Maximum RPM cannot be higher than 15000 RPM.')
        if ve_mode == 'table':
            results = FullRangeSweep(rpmMin, rpmMax, displacement, ve_mode, ve_table=ve_vs_rpm)
        else:
            results = FullRangeSweep(rpmMin, rpmMax, displacement, ve_mode, constant_ve=ve)
        df = pd.DataFrame(results, columns=['Engine Speed (RPM)', 'Throttle', 'Torque (Nm)', 'Power (kW)', 'Horsepower', 'Air Flow(g/s)', 'Fuel Flow(g/s)', 'CO2(g/s)', 'CO(g/s)', 'NOx(g/s)', 'HC(g/s)'])
        export_results_to_csv(df)
        rpm_vs_plots(df)
        sys.exit()
    elif testMode == '4':
        print("Exiting program.")
        sys.exit()
    else:
        print("Invalid input. Please enter 1, 2, or 3.")
