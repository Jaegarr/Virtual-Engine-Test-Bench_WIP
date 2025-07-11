import math
def calculate_torque(rpm, throttle):
    """
    Simple torque model:
    Torque = a * throttle * rpm - b * rpm^2
    """
    a = 0.05
    b = 0.00001

    torque = a * throttle * rpm - b * rpm**2
    if torque < 0:
        torque = 0
    return torque
def calculate_power(torque, rpm):
    power = (torque*rpm*2*math.pi/60)/1000 # In kW