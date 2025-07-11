from engine_model import calculate_power, calculate_torque
while True:
    try:
        print('Enter throttle position:')
        throttle = float(input())
        if throttle < 0.0 or throttle > 1.0:
            print('Throttle position must be between 0 and 1. Please enter a throttle position:')
            continue
        break
    except:
        print('Invalid input. Throttle position must be between 0 and 1')
'''
while True:
    try:
        print('Enter engine RPM:')
        rpm = float(input())
        if rpm < 0.0 or rpm > 15000.0:
            print('Engine RPM must be between 0 and 15000. Please enteR engine RPM:')
            continue
        break
    except:
        print('Invalid input. Engine RPM must be between 0 and 15000')
'''
for rpm in range(1000,8000,500): #RPM Sweep
     t = calculate_torque(rpm, throttle)
     p = calculate_power(rpm, t)
     print(f'Your torque at {rpm} RPM is {t} Nm.') 
     print(f'Your power at {rpm} RPM is {p} kW.')      
