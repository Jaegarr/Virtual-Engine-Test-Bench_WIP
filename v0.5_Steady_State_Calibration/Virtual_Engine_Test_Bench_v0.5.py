from Test_Modes import FullRangeSweep, WideOpenThrottle, SingleRun
from Reporting import export_results_to_csv,rpm_vs_plots, emission_plots
import sys
import pandas as pd
import Calibration as cal
pd.set_option('display.float_format', '{:.3f}'.format)
#%%Engine Displacement Input
while True:
    try:
        print('Enter engine displacement in Liters:')
        displacement = float(input())
        if displacement < 0.0 or displacement > 20.0: # Assuming vehicle is passenger & light/medium off-highway vehicle
            print('Engine displacement must be between 0 and 20. Please enter engine displacement:')
            continue
        break
    except:
        print('Invalid input. Engine displacement must be between 0 and 20.')
#%% VE Input
while True:
    print("Choose VE input method:")
    print("1. Enter constant VE")
    print("2. Load VE table (CSV)")
    print("3. Exit the program")
    choice = input("Enter 1, 2 or 3: ").strip()
    if choice == '1':
        while True:
            try:
                print('Enter Volumetric Efficiency(VE):')
                ve = float(input())
                if ve < 0.6 or ve > 1.8:
                    print('VE must be between 0.6 and 1.8.')
                    continue
                ve_mode = 'constant'
                break
            except:
                print('Invalid input. VE must be between 0.6 and 1.8')
        break  # break outer loop after success
    elif choice == '2':
        file_path = input('Enter VE table CSV path (leave empty to use default): ').strip()
        ve_vs_rpm =  pd.read_csv(file_path if file_path else 'C:/Users/berke/OneDrive/Masaüstü/GitHub/Virtual-Engine-Test-Bench/v0.5_Steady_State_Calibration/Nissan_350Z_VE.csv', index_col=0)
        if ve_vs_rpm is None:
            print('❌ Failed to load VE data. Try again.')
            continue  # goes back to asking VE input method
        else:
            ve_mode = 'table'
            break  # break outer loop after loading
    elif choice == '3':
        print('Exiting the program.')
        sys.exit()
    else:
        print('Invalid choice, enter 1, 2 or 3.')
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
            df = pd.DataFrame(results, columns=['Engine Speed (RPM)', 'Throttle', 'Torque (Nm)', 'Power (kW)', 'Horsepower',  'CO2(g/s)', 'CO(g/s)', 'NOx(g/s)', 'HC(g/s)'])
            export_results_to_csv(df)
            sys.exit()
    elif testMode == '2':
        print("You selected Full Throttle Response")
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
            results = WideOpenThrottle(rpmMin, rpmMax, displacement, ve_mode, ve_table=ve_vs_rpm)
        else:
            results = WideOpenThrottle(rpmMin, rpmMax, displacement, ve_mode, constant_ve=ve)
        df = pd.DataFrame(results, columns=['Engine Speed (RPM)', 'Throttle', 'Torque (Nm)', 'Power (kW)', 'Horsepower',  'CO2(g/s)', 'CO(g/s)', 'NOx(g/s)', 'HC(g/s)'])
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
        df = pd.DataFrame(results, columns=['Engine Speed (RPM)', 'Throttle', 'Torque (Nm)', 'Power (kW)', 'Horsepower',  'CO2(g/s)', 'CO(g/s)', 'NOx(g/s)', 'HC(g/s)'])
        export_results_to_csv(df)
        rpm_vs_plots(df)
        sys.exit()
    elif testMode == '4':
        print("Exiting program.")
        sys.exit()
    else:
        print("Invalid input. Please enter 1, 2, or 3.")
