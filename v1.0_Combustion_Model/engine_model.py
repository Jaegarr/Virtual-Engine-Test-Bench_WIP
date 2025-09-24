import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Calibration as cal
from Engine_Database import EngineSpec
def combustion_Wiebe( spec: EngineSpec,rpm , throttle, ve, # Inputs
                     LHV = 44E6, gas_constant = 287, T_ivc = 330, cv_J = 750, # Fuel / Thermodynamics 
                     a = 5, m = 2, combustion_efficiency = 0.98, n_poly_compression = 1.34, n_poly_expansion = 1.26, # Wiebe Parameters
                     plot = True, return_dic = False): # I/O
    # CRANK ANGLE GRID
    crank_angle = np.linspace(-np.pi, np.pi, 1441)
    dtheta = crank_angle[1] - crank_angle[0]
    # VALVE & BURN TIMINGS
    ivo_rad = np.deg2rad(130.0)
    ivc_rad = np.deg2rad(-110.0)
    soc_rad = np.deg2rad(-10.0)
    eoc_rad = np.deg2rad(25.0)
    evo_rad = np.deg2rad(110.0)
    evc_rad = np.deg2rad(-160.0)
    i_ivo = int(np.argmin(np.abs(crank_angle - ivo_rad))) 
    i_ivc = int(np.argmin(np.abs(crank_angle - ivc_rad)))
    i_soc = int(np.argmin(np.abs(crank_angle - soc_rad)))
    i_eoc = int(np.argmin(np.abs(crank_angle - eoc_rad))) 
    i_evo = int(np.argmin(np.abs(crank_angle - evo_rad)))
    i_evc = int(np.argmin(np.abs(crank_angle - evc_rad))) 
    delta = eoc_rad - soc_rad
    # cv(T) MAPS
    T_knots = np.array([300, 600, 1000, 1500, 2000, 2500, 3000, 3500])
    cv_u_knots = np.array([718, 740, 820,  900,  960, 1000, 1030, 1050])   # unburned (air-ish)
    cv_b_knots = np.array([740, 780, 880, 1000, 1100, 1150, 1200, 1250])   # burned products
    def cv_unburned(T):
        T = np.clip(T, 300.0, 3500.0)
        return np.interp(T, T_knots, cv_u_knots)
    def cv_burned(T):
        T = np.clip(T, 300.0, 3500.0)
        return np.interp(T, T_knots, cv_b_knots)
    def cv_mix(T, mfb):
        return (1.0 - mfb) * cv_unburned(T) + mfb * cv_burned(T) # linear blend by mass-fraction-burned (0..1)
    # CA10/50/90
    def ca_at_mfb(y):
        x = (-np.log(1.0 - y) / a)**(1.0 / (m + 1.0))
        return soc_rad + x * delta
    ca10_rad = ca_at_mfb(0.10); ca50_rad = ca_at_mfb(0.50); ca90_rad = ca_at_mfb(0.90)
    ca10_deg, ca50_deg, ca90_deg = map(np.degrees, (ca10_rad, ca50_rad, ca90_rad))
    idx10 = int(np.argmin(np.abs(crank_angle - ca10_rad)))
    idx50 = int(np.argmin(np.abs(crank_angle - ca50_rad)))
    idx90 = int(np.argmin(np.abs(crank_angle - ca90_rad)))
    # GEOMETRY
    V_displacement = np.pi * (spec.bore_m**2 / 4) * spec.stroke_m
    V_clearance = V_displacement / (spec.compression_ratio - 1)
    crank_radius = spec.stroke_m / 2
    crossSec = np.pi * spec.bore_m**2 / 4
    # POSITION
    piston_pos = crank_radius * (1 - np.cos(crank_angle)) + crank_radius**2 / (2 * spec.conrod_m) * (1 - np.cos(2 * crank_angle))
    V = V_clearance + crossSec * piston_pos
    dV_dtheta = np.gradient(V, crank_angle)
    # TRAPPED MASS
    p_ivc = 20e3 + throttle * (100e3 - 20e3)  # Pa
    rho_ivc =  p_ivc / (gas_constant * T_ivc) # kg/m3
    mAirpercycle = rho_ivc * V_displacement * ve  # kg per cycle
    mAirpersec = mAirpercycle * spec.n_cylinder * rpm / 120 # All cylinders
    mFuelpercycle = mAirpercycle / cal.get_target_AFR(rpm) # kg per cycle
    mFuelpersec = mAirpersec / cal.get_target_AFR(rpm)
    m_trapped = mAirpercycle + mFuelpercycle
    # COMPRESSION STROKE
    V_compression = V[i_ivc:i_soc+1]
    P_compression = (p_ivc * (V[i_ivc]/V_compression) ** n_poly_compression)
    T_compression = (P_compression * V_compression) / (m_trapped * gas_constant)
    # TOTAL ENERGY RELEASE
    Q_tot = mFuelpercycle * combustion_efficiency * LHV  # keep your loss knob off for now
    # COMBUSTION
    P_current = P_compression[-1]
    T_current = T_compression[-1]
    P_combustion, T_combustion, mfb_list = [], [], []
    for i in range(i_soc, i_eoc+1):
        theta = crank_angle[i]
        x = (theta - soc_rad) / delta
        x = np.clip(x, 0.0, 1.0)
        mfb = 1.0 - np.exp(-a * x**(m+1))
        dmfb_dtheta = a * (m+1) * x**m * np.exp(-a * x**(m+1)) / delta
        dQ_chem = Q_tot * dmfb_dtheta * dtheta
        dU = dQ_chem * 0.9 - P_current * (V[i+1] - V[i])  # dU = dQ - dW
        cv_loc = cv_mix(T_current, mfb)
        dT_combustion = dU / (m_trapped * cv_loc)
        V_current = V[i+1]
        T_current += dT_combustion
        P_current = m_trapped * gas_constant * T_current / V_current
        T_combustion.append(T_current)
        P_combustion.append(P_current)
        mfb_list.append(mfb)
    # EXPANSION STROKE
    P_eoc = P_combustion[-1]
    V_eoc = V[i_eoc]
    V_expansion = V[i_eoc:i_evo+1]
    P_expansion = P_eoc * (V_eoc / V_expansion) ** n_poly_expansion
    T_expansion = P_expansion * V_expansion / (m_trapped * gas_constant)
    # BLOWDOWN
    blowdown_rad = crank_angle[i_evo:i_ivo+1]
    V_blowdown = V[i_evo:i_ivo+1]
    P_exhaust = 1.05e5
    T_exhaust_target = 1150.0
    V_ivo = V[i_ivo]
    m_target = P_exhaust * V_ivo / (gas_constant * T_exhaust_target)
    P_evo = P_expansion[-1]
    T_evo = T_expansion[-1]
    m_evo = m_trapped
    omega = rpm * 2*np.pi / 60.0
    delta_m = m_evo - m_target
    den = max(ivo_rad - evo_rad, 1e-12)
    alpha = np.clip((blowdown_rad - evo_rad) / den, 0.0, 1.0)
    beta = 0.30
    S = (1.0 - np.exp(-alpha / beta)) / (1.0 - np.exp(-1.0 / beta))
    m_bd = m_evo - delta_m * S
    m_bd = np.minimum.accumulate(m_bd)
    P_bd = np.empty_like(blowdown_rad)
    T_bd = np.empty_like(blowdown_rad)
    T_bd[0] = T_evo
    P_bd[0] = m_bd[0] * gas_constant * T_bd[0] / V_blowdown[0]
    for i in range(len(blowdown_rad) - 1):
        dtheta_k = blowdown_rad[i+1] - blowdown_rad[i]
        dt_k = dtheta_k / omega
        dVdtheta_k = (V_blowdown[i+1] - V_blowdown[i]) / dtheta_k
        dVdt_k = dVdtheta_k * omega
        dm_dtheta_k = (m_bd[i+1] - m_bd[i]) / dtheta_k
        mdot_out_k = -dm_dtheta_k * omega
        P_k = m_bd[i] * gas_constant * T_bd[i] / V_blowdown[i]
        cv_bd = cv_burned(T_bd[i])
        dTdt_k = (-P_k * dVdt_k - gas_constant * T_bd[i] * mdot_out_k) / (m_bd[i] * cv_bd)
        T_bd[i+1] = max(300.0, T_bd[i] + dTdt_k * dt_k)
        P_bd[i+1] = m_bd[i+1] * gas_constant * T_bd[i+1] / V_blowdown[i+1]
    P_blowdown, T_blowdown = P_bd, T_bd
    # DATA STORAGE
    df = pd.DataFrame({
        'Crank Angle (deg)': np.degrees(crank_angle),
        'Volume (m3)': V,
        'dVdtheta (m3/rad)': dV_dtheta,
        'Pressure (bar)': np.nan,
        'Temperature (K)':  np.nan,
        'Mass Fraction Burned': np.nan
    })
    df.loc[i_ivc:i_soc, 'Pressure (bar)'] = P_compression / 1e5
    df.loc[i_soc:i_eoc, 'Pressure (bar)'] = np.array(P_combustion) / 1e5
    df.loc[i_eoc:i_evo, 'Pressure (bar)'] = P_expansion / 1e5
    df.loc[i_evo:i_ivo, 'Pressure (bar)'] = P_blowdown / 1e5
    df.loc[i_ivc:i_soc, 'Temperature (K)']  = T_compression
    df.loc[i_soc:i_eoc, 'Temperature (K)']  = np.array(T_combustion)
    df.loc[i_eoc:i_evo, 'Temperature (K)']  = T_expansion
    df.loc[i_evo:i_ivo, 'Temperature (K)']  = T_blowdown
    df.loc[i_soc:i_eoc, 'Mass Fraction Burned'] = np.array(mfb_list)
    # p–V + IMEP 
    P_pa = (df['Pressure (bar)'].to_numpy() * 1e5)
    V_m3 = df['Volume (m3)'].to_numpy()
    mask = ~np.isnan(P_pa)
    W_cyl = np.trapezoid(P_pa[mask], V_m3[mask])
    imep_gross = W_cyl / V_displacement
    mean_piston = 2 * spec.stroke_m * rpm / 60
    fmep = (0.6 + 0.00050 * mean_piston + 0.00550 * (mean_piston) ** 2) * 1e5
    pmep = (0.03 + 0.000035 * rpm) * 1e5 
    bmep = imep_gross - fmep - pmep
    Vd_total = V_displacement * spec.n_cylinder
    Torque_Nm = bmep * Vd_total / (4*np.pi)      # 4π for 4-stroke
    omega = rpm * 2*np.pi / 60.0
    Power_kW = Torque_Nm * omega / 1000.0
    bsfc = mFuelpersec * 3600 / Power_kW
    # PLOTTING
    if plot:
        plt.figure()
        plt.plot(V_m3[mask], P_pa[mask]/1e5)
        plt.xlabel('Volume [m³]'); plt.ylabel('Pressure [bar]')
        plt.title('p–V Loop (single cylinder)'); plt.grid(True); plt.show()
        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(df['Crank Angle (deg)'], df['Pressure (bar)'])
        for ca_deg, label in [(np.degrees(ca10_rad),'CA10'), (np.degrees(ca50_rad),'CA50'), (np.degrees(ca90_rad),'CA90')]:
            plt.axvline(ca_deg, ls='--', lw=1, c='k', alpha=0.7)
            ymax = plt.gca().get_ylim()[1]; plt.text(ca_deg, 0.95*ymax, label, rotation=90, va='top', ha='right', fontsize=9)
        plt.ylabel('Pressure [bar]'); plt.grid(True)

        plt.subplot(3,1,2)
        plt.plot(df['Crank Angle (deg)'], df['Temperature (K)'])
        for ca_deg in (np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)):
            plt.axvline(ca_deg, ls='--', lw=1, c='k', alpha=0.5)
        plt.ylabel('Temperature [K]'); plt.grid(True)

        plt.subplot(3,1,3)
        plt.plot(df['Crank Angle (deg)'], df['Mass Fraction Burned'])
        for ca_deg in (np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)):
            plt.axvline(ca_deg, ls='--', lw=1, c='k', alpha=0.5)
        for y in (0.10, 0.50, 0.90): plt.axhline(y, ls=':', lw=1, c='gray', alpha=0.6)
        plt.plot([np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)], [0.10, 0.50, 0.90], 'ko', ms=4)
        plt.xlabel('Crank Angle [deg]'); plt.ylabel('MFB [-]'); plt.grid(True)
        plt.tight_layout(); plt.show()
        return
    #RESULTS
    if return_dic:
        out = {
        "imep_gross_pa": imep_gross,
        "bmep_pa": bmep,
        "fmep_pa": fmep,
        "pmep_pa": pmep,
        "torque_nm": Torque_Nm,
        "power_kw": Power_kW,
        "m_air_per_cycle": mAirpercycle,   # per cyl, per cycle
        "m_fuel_per_cycle": mFuelpercycle, # per cyl, per cycle
        "ca10_deg": ca10_deg,
        "ca50_deg": ca50_deg,
        "ca90_deg": ca90_deg,
        "pmax_bar": np.nanmax(df['Pressure (bar)'].to_numpy()),
        "tmax_k":   np.nanmax(df['Temperature (K)'].to_numpy()),
    }
        return out
def estimate_Emissions(mDotFuel, AFR, eff):
    """
    Rough estimate of emissions based on fuel mass flow and AFR.
    fuel_mass_flow: kg/s
    afr: actual AFR
    combustion_eff: assumed combustion efficiency (0-1)
    Returns CO2, CO, NOx and HC in g/s
    """
    # Empirical Scaling Factors
    k_co = 0.2
    k_nox = 0.08
    k_thc = 0.03
    mDotFuel = mDotFuel * 1000
    lambda_val = AFR / 14.7
    mDotco2 = mDotFuel * 3.09 # CO2: 3.09 g CO2 per g fuel burned
    mDotco = mDotFuel * k_co * max(0, abs(1.2 - lambda_val)) # CO: Rises when rich (lambda < 1.2)
    mDotnox = mDotFuel * k_nox * max(0, 1 - abs(lambda_val - 1)) # NOx: peaks near stoichiometric, lower when far rich/lean
    mDotthc = mDotFuel * k_thc * (1 - eff) # THC mainly from incomplete combustion
    emissions_gps = [mDotco2, mDotco, mDotnox, mDotthc] # g/s
    return emissions_gps
def estimate_emissions(mDotFuel, AFR, comb_eff, load_frac=0.6, ei_co2_g_per_kg=3090.0):
    """
    Estimate engine-out emissions as g/s from fuel flow, AFR, and a load proxy.

    Parameters
    ----------
    mDotFuel_kgps : float
        Fuel mass flow [kg/s]
    AFR : float
        Actual air-fuel ratio (mass-based). Stoich gasoline ≈ 14.7.
    comb_eff : float
        Combustion efficiency (0..1). Influences HC primarily.
    load_frac : float
        0..1 proxy for load/BMEP (affects NOx amplitude).
    ei_co2_g_per_kg : float
        Emission index for CO2 [g/kg fuel]. ~3090 for gasoline.

    Returns
    -------
    dict : {'CO2': g/s, 'CO': g/s, 'NOx': g/s, 'HC': g/s}
        Engine-out (pre-catalyst) emission rates.
    """
    lam = AFR / 14.7
    lam = max(0.5, min(1.6, lam))
    # --- CO (g/kg fuel) ---
    # Very low when lean; rises rapidly rich of stoich.
    # Smooth curve: quadratic increase as lambda goes below 1.
    # Typical hot engine-out EI_CO at λ≈0.9 can be O(100 g/kg), lean ~<5 g/kg.
    if lam >= 1.0:
        EI_CO = 3.0 * (1 + 4.0*(lam - 1.0))  # slightly increases if very lean due to misfire risk
    else:
        EI_CO = 5.0 + 800.0*(1.0 - lam)**2   # rich penalty
    EI_CO = min(EI_CO, 400.0)  # cap to avoid extremes
    # --- HC (g/kg fuel) ---
    # Rises rich (over-fuel/quench) and very lean (misfire), plus incomplete combustion.
    rich_term = 200.0*(max(0.0, 1.0 - lam))**2
    lean_term = 120.0*(max(0.0, lam - 1.15))**2
    incomp_term = 50.0*(1.0 - max(0.0, min(1.0, comb_eff)))
    EI_HC = 2.0 + rich_term + lean_term + incomp_term
    EI_HC = min(EI_HC, 300.0)
    # --- NOx (g/kg fuel) ---
    # Peak slightly lean of stoich; scale with load (temperature).
    # Use a Gaussian around lambda≈1.05 with width ~0.08–0.10.
    lam_peak = 1.05
    sigma = 0.09
    peak_noX = 18.0 * (load_frac**0.7)  # higher load → more NOx
    EI_NOx = peak_noX * np.exp(-0.5*((lam - lam_peak)/sigma)**2)
    # Mild lean/rich suppression already handled by Gaussian
    # --- CO2 (g/kg fuel) ---
    EI_CO2 = ei_co2_g_per_kg  # essentially fixed by fuel chemistry
    # Convert EI [g/kg fuel] to g/s using fuel flow [kg/s]
    gps_CO2 = EI_CO2 * mDotFuel
    gps_CO  = EI_CO  * mDotFuel
    gps_NOx = EI_NOx * mDotFuel
    gps_HC  = EI_HC  * mDotFuel

    return {'CO2': gps_CO2, 'CO': gps_CO, 'NOx': gps_NOx, 'HC': gps_HC}
