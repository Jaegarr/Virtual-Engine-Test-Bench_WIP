import math
import Calibration as cal
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
    displacement_m3 = displacement_l/1e3
    mdot = displacement_m3*ve*rpm*rho/(2*60)
    return mdot
def calculate_torque(rpm, throttle, mdotAir, displacement_l, LHV = 44e6, eff = 0.3):
    mdotFuel = mdotAir*throttle/cal.get_target_lambda(rpm)
    gross_torque = mdotFuel*LHV*eff/(rpm*2*math.pi/60) # In Nm
    fmep = 0.25 + 0.02*rpm/1000 + 0.03*(rpm/1000)**2
    displacement_m3 = displacement_l/1e3
    #FMEP
    fmep_pa = fmep*1e5
    torque_fmep = fmep_pa * displacement_m3/(4*math.pi)
    #PMEP
    pmep = 0.02 + 0.00001*rpm
    pmep_pa = pmep*1e5
    torque_pmep = pmep_pa*displacement_m3/(4*math.pi)
    #Net Torque
    torque_net = gross_torque - (torque_fmep + torque_pmep)
    return max(torque_net,0) # I am not confident with myself sometimes
def calculate_power(rpm, torque):
    power = (torque*rpm*2*math.pi/60)/1000 # In kW
    return power
def calculate_horsePower(rpm,torque):
    horsePower = (torque*rpm*2*math.pi/60)/745.7
    return horsePower

