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