import engine_model
def ThrottleRPMSweep(RPM_min,RPM_max):
    results=[]
    for rpm in range(RPM_min,RPM_max+1,500):
        t25 = engine_model.calculate_torque(rpm,0.25)
        t50 = engine_model.calculate_torque(rpm,0.5)
        t75 = engine_model.calculate_torque(rpm,0.75)
        t100 = engine_model.calculate_torque(rpm,1)
        row = [rpm, t25, t50, t75, t100]
        results.append(row)
    return results
ThrottleRPMSweep(1000,8000)