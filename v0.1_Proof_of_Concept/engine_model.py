import math
def calculate_torque(rpm, throttle):
    a = 0.05
    b = 0.00001
    #a and b are arbitrary
    torque = a * throttle * rpm - b * rpm**2 #In N
    return max(torque,0)
def calculate_power(rpm, torque):
    power = (torque*rpm*2*math.pi/60)/1000 # In kW
    return power