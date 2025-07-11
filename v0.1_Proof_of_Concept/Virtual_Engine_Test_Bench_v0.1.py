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
            print('VE must be between 0.6 and 1.2. Please enteR VE:')
            continue
        break
    except:
        print('Invalid input. VE must be between 0.6 and 1.2')
'''
for ve in range(1000,8000,500): #RPM Sweep
     t = calculate_torque(ve, displacement)
     p = calculate_power(ve, t)
     print(f'Your torque at {ve} RPM is {t} Nm.') 
     print(f'Your power at {ve} RPM is {p} kW.') 
'''     
results = ThrottleRPMSweep(1000, 8000, displacement, ve)
df = pd.DataFrame(results, columns=['RPM', 'Throttle', 'Torque (Nm)', 'Power (kW)'])
print(df)