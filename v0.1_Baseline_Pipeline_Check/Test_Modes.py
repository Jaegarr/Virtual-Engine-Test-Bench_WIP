from engine_model import calculate_air_mass_flow, calculate_power, calculate_torque, calculate_horsePower
def SingleRun(rpm, displacement_l, ve):
    mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
    results=[]
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for throttle in throttles:
        t = calculate_torque(rpm, throttle, mdotAir)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm,t)
        row = [rpm, throttle, t, p, hp]
        results.append(row)
    return results
def FullThrottleResponse(RPM_min, RPM_max, displacement_l, ve):
    results = []
    throttle = 1.0
    for rpm in range(RPM_min,RPM_max+1,100):
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        t = calculate_torque(rpm, throttle, mdotAir)
        p = calculate_power(rpm, t)
        hp = calculate_horsePower(rpm,t)
        row = [rpm, throttle, t, p, hp]
        results.append(row)
    return results
def ThrottleRPMSweep(RPM_min, RPM_max, displacement_l, ve):
    results=[]
    throttles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for rpm in range(RPM_min,RPM_max+1,100):
        mdotAir = calculate_air_mass_flow(rpm, displacement_l, ve)
        for throttle in throttles:
            t = calculate_torque(rpm, throttle, mdotAir)
            p = calculate_power(rpm, t)
            hp = calculate_horsePower(rpm,t)
            row = [rpm, throttle, t, p, hp]
            results.append(row)
    return results
