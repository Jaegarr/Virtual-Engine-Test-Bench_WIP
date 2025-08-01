from engine_model import calculate_air_mass_flow, calculate_power, calculate_torque, calculate_horsePower
import Calibration as cal
def SingleRun(rpm, displacement_l, ve):
    mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
    results=[]
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for throttle in throttles:
        t = calculate_torque(rpm, throttle, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm,t)
        row = [rpm, throttle, t, p, hp]
        results.append(row)
    return results
def FullThrottleResponse(RPM_min, RPM_max, displacement_l, ve_mode, ve_vs_rpm=None, constant_ve=None):
    results = []
    throttle = 1.0
    for rpm in range(RPM_min,RPM_max+1,100):
        if ve == 'table':
            ve = cal.get_ve_from_table(rpm,ve_vs_rpm)
        else:
            ve = constant_ve
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t = calculate_torque(rpm, throttle, mdotAir, displacement_l)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm,t)
        row = [rpm, throttle, t, p, hp]
        results.append(row)
    return results
def FullRangeSweep(RPM_min, RPM_max, displacement_l, ve):
    results=[]
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for rpm in range(RPM_min,RPM_max+1,100):
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        for throttle in throttles:
            t = calculate_torque(rpm, throttle, mdotAir, displacement_l)
            p = calculate_power(rpm, t)
            hp = calculate_horsePower(rpm,t)
            row = [rpm, throttle, t, p, hp]
            results.append(row)
    return results
