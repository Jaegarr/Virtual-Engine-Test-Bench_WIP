from Test_Sweep import ThrottleRPMSweep
import pandas as pd
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
while True:
    try:
        print('Enter Volumetric Efficiency(VE):')
        ve = float(input())
        if ve < 0.6 or ve > 1.2: # Assuming NA or mild turbo
            print('VE must be between 0.6 and 1.2. Please enter VE:')
            continue
        break
    except:
        print('Invalid input. VE must be between 0.6 and 1.2')
while True:
    try:
        print('Enter minimum RPM:')
        rpmMin = int(input())
        if rpmMin < 800: # Assuming NA or mild turbo
            print('Minimum RPM must be at least 800 RPM. Please enter minimum RPM:')
            continue
        break
    except:
        print('Invalid input. Minimum RPM must be at least 800 RPM.')
while True:
    try:
        print('Enter maximum RPM:')
        rpmMax = int(input())
        if ve > 15000: # Assuming NA or mild turbo
            print('Maximum RPM cannot be higher than 15000 RPM. Please enter maximum RPM:')
            continue
        break
    except:
        print('Invalid input. Maximum RPM cannot be higher than 15000 RPM.')
results = ThrottleRPMSweep(rpmMin, rpmMax, displacement, ve)
df = pd.DataFrame(results, columns=['RPM', 'Throttle', 'Torque (Nm)', 'Power (kW)'])
print(df)