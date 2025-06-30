import Test_Sweep
'''
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
for rpm in range(1000,8000,500): #RPM Sweep
     x = calculate_torque(rpm, throttle)
     print(f'Your torque at {rpm} RPM is {x}')
'''
result = Test_Sweep.ThrottleRPMSweep(1000,8000)
for row in result:
    print(row)