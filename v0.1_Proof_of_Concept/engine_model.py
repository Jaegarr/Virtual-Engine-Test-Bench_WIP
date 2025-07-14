import math
def calculate_air_mass_flow(rpm, displacement_l, ve, rho = 1.22588):   
    '''
    Calculate the air mass flow rate (kg/s) into a 4-stroke engine.

    Parameters:
        rpm (float): Engine speed in revolutions per minute
        displacement_l (float): Engine displacement in liters
        VE (float): Volumetric efficiency (0-1.2)
        air_density (float): Air density in kg/m³ (default 1.225)
    Returns:
        float: Air mass flow rate in kg/s
    '''
    displacement_m = displacement_l/1e3
    mdot = displacement_m*ve*rpm*rho/(2*60)
    return mdot
def calculate_torque(rpm, throttle, mdotAir, AFR = 14.7, LHV = 44e6, eff = 0.3):
    mdotFuel = mdotAir*throttle/AFR
    torque = mdotFuel*LHV*eff/(rpm*2*math.pi/60) # In Nm
    return torque
def calculate_power(rpm, torque):
    power = (torque*rpm*2*math.pi/60)/1000 # In kW
    return power