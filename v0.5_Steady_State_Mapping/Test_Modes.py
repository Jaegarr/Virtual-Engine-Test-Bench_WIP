from engine_model import calculate_air_mass_flow, calculate_power, calculate_torque, calculate_horsePower
import Calibration as cal

def SingleRun(rpm, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Perform a single engine speed simulation across a range of throttle positions.

    Parameters:
        rpm (int or float): Engine speed in revolutions per minute.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle, Torque (Nm), Power (kW), Horsepower].
                       The list covers throttle values from 0.1 to 1.0 in 0.1 increments.
    """
    results = []
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if ve_mode == 'table':
        ve_list = cal.get_ve_from_table(rpm, throttles, ve_table)
    else:
        ve_list = [constant_ve] * len(throttles)
    for throttle, ve in zip(throttles, ve_list):
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t, emissions = calculate_torque(rpm, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm, t)
        row = [rpm, throttle, t, p, hp] + emissions
        results.append(row)
    return results
def FullThrottleResponse(RPM_min, RPM_max, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Simulate engine performance at full throttle (100%) over a specified RPM range.

    Parameters:
        RPM_min (int): Minimum RPM for the simulation.
        RPM_max (int): Maximum RPM for the simulation.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle=1.0, Torque (Nm), Power (kW), Horsepower].
                       Covers RPM range in steps of 100 RPM.
    """
    results = []
    throttle = 1.0
    for rpm in range(RPM_min, RPM_max + 1, 100):
        if ve_mode == 'table':
            ve = cal.get_ve_from_table(rpm, throttle, ve_table)
        else:
            ve = constant_ve
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t, emissions = calculate_torque(rpm, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm, t)
        row = [rpm, throttle, t, p, hp] + emissions
        results.append(row)
    return results
def FullRangeSweep(RPM_min, RPM_max, displacement_l, ve_mode, ve_table=None, constant_ve=None):
    """
    Perform a full simulation sweep across RPM and throttle range.

    Parameters:
        RPM_min (int): Minimum RPM for the sweep.
        RPM_max (int): Maximum RPM for the sweep.
        displacement_l (float): Engine displacement in liters.
        ve_mode (str): 'table' to use VE table interpolation, 'constant' for fixed VE.
        ve_table (pandas.DataFrame, optional): VE table used if ve_mode=='table'.
        constant_ve (float, optional): Constant volumetric efficiency used if ve_mode=='constant'.

    Returns:
        list of lists: Each inner list contains [RPM, Throttle, Torque (Nm), Power (kW), Horsepower].
                       Covers RPM range in steps of 100 RPM and throttle from 0.1 to 1.0 in 0.1 increments.
    """
    results = []
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for rpm in range(RPM_min, RPM_max + 1, 100):
        if ve_mode == 'table':
            ve_list = cal.get_ve_from_table(rpm, throttles, ve_table)
        else:
            ve_list = [constant_ve] * len(throttles)
        for throttle, ve in zip(throttles, ve_list):
            mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
            t, emissions = calculate_torque(rpm, mdotAir, displacement_l)
            p = calculate_power(rpm, t)
            hp = calculate_horsePower(rpm, t)
            row = [rpm, throttle, t, p, hp] + emissions
            results.append(row)
    return results