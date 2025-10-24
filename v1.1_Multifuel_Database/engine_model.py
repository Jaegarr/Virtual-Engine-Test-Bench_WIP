import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Calibration as cal
from Engine_Database import EngineSpec
from Fuel_Database import FuelSpec, Fuels
#%% FLAME SPEED
def laminar_speed(
    fuel: FuelSpec,
    T_u_K: float,
    p_u_Pa: float,
    phi: float,
    Sref_mps: float = None,
    alpha_T: float = 2.0,     # T exponent (generic)
    beta_p: float = -0.25,    # p exponent (generic)
    phi_peak: float = 1.1,    # where S_L peaks (gasolineish)
    phi_width: float = 0.35   # width of quadratic around peak
) -> float:
    """
    Generic S_L correlation built around S_L at 300K, 1 atm.
    S_L ≈ S_L,300 * (T/300)^alpha * (p/1atm)^beta * phi-shape
    """
    S0 = fuel.S_L_300K_m_per_s if Sref_mps is None else Sref_mps
    T_term = (T_u_K / 300.0)**alpha_T
    p_term = (p_u_Pa / 101325.0)**beta_p
    # simple parabola in phi around phi_peak: ~1 - ((phi-phi_peak)/width)^2
    phi_shape = max(0.15, 1.0 - ((phi - phi_peak)/phi_width)**2)
    return max(0.02, S0 * T_term * p_term * phi_shape)

def turbulent_speed(
    S_L: float,
    k_u: float,
    ell_t_m: float,
    nu_u_m2s: float = 1.7e-5,
    Ct: float = 1.5,
    n: float = 1.0,
    m: float = 0.3
) -> float:
    """
    Minimal Bradley-style wrinkled-flame model:
      u' = sqrt(2/3 k)
      S_T = S_L * (1 + Ct * (u'/S_L)^n * (l_t/delta_L)^m)
    With m=0 this reduces to S_T ≈ S_L(1 + Ct u'/S_L).
    """
    u_prime = np.sqrt(max(0.0, 2.0/3.0 * k_u))
    # laminar flame thickness ~ thermal diffusivity / S_L (rough approx)
    alpha_th = 2.0e-5
    delta_L = max(5.0e-5, alpha_th / max(0.02, S_L))  # clamp thickness ≥0.2 mm
    scale = (u_prime / max(0.02, S_L))**n * ( (ell_t_m / delta_L)**m if m != 0 else 1.0 )
    S_T = S_L * (1.0 + Ct * scale)
    return float(np.clip(S_T, max(S_L*1.05, 3.0), 40.0))
def combustion_Wiebe( spec: EngineSpec,
                     rpm , throttle, ve,  # Inputs
                     fuel: FuelSpec,
                     k_mean: float = 5.0,        # [m^2/s^2] from CFD or guess
                     T_u_mean: float = 330.0,    # unburned-gas mean T (CFD/OpenFOAM)
                     t_length_mean: float = None,      # turbulence length scale
                     gas_constant = 287,         # R
                     T_ivc = 330,
                     a = 5, m = 2,               # Wiebe shape
                     combustion_efficiency = 0.98,
                     n_poly_compression = 1.34, n_poly_expansion = 1.26,
                     plot = True, return_dic = False):  # I/O
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
    i_evo = int(np.argmin(np.abs(crank_angle - evo_rad)))
    i_evc = int(np.argmin(np.abs(crank_angle - evc_rad)))
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
        # linear blend by mass-fraction-burned (0..1)
        return (1.0 - mfb) * cv_unburned(T) + mfb * cv_burned(T)
    # GEOMETRY
    V_displacement = np.pi * (spec.bore_m**2 / 4) * spec.stroke_m
    V_clearance = V_displacement / (spec.compression_ratio - 1)
    crank_radius = spec.stroke_m / 2
    crossSec = np.pi * spec.bore_m**2 / 4
    # POSITION
    piston_pos = crank_radius * (1 - np.cos(crank_angle)) + \
                 crank_radius**2 / (2 * spec.conrod_m) * (1 - np.cos(2 * crank_angle))
    V = V_clearance + crossSec * piston_pos
    dV_dtheta = np.gradient(V, crank_angle)
    # TRAPPED MASS
    p_ivc = 20e3 + throttle * (100e3 - 20e3)                       # Pa
    rho_ivc =  p_ivc / (gas_constant * T_ivc)                      # kg/m3
    mAirpercycle = rho_ivc * V[i_ivc] * ve                         # kg/cycle
    afr_actual = cal.get_target_AFR(rpm, fuel=fuel)
    mAirpersec = mAirpercycle * spec.n_cylinder * rpm / 120        # All cylinders
    mFuelpercycle = mAirpercycle / afr_actual                      # kg/cycle
    mFuelpersec = mFuelpercycle * spec.n_cylinder * rpm / 120      # All cylinders
    m_trapped = mAirpercycle + mFuelpercycle                       # kg/cycle
    # COMPRESSION STROKE
    i_soc = int(np.argmin(np.abs(crank_angle - soc_rad)))
    V_compression = V[i_ivc:i_soc+1]
    P_compression = (p_ivc * (V[i_ivc]/V_compression) ** n_poly_compression)
    T_compression = (P_compression * V_compression) / (m_trapped * gas_constant)
    # TOTAL ENERGY RELEASE
    Q_tot = mFuelpercycle * combustion_efficiency * fuel.LHV_MJ_per_kg * 1e6
    # COMBUSTION INITIAL STATE
    P_current = P_compression[-1]
    T_current = T_compression[-1]
    p_soc = P_compression[-1]
    T_soc = T_compression[-1]
    phi = max(0.0, min(2.0, fuel.AFR_stoich / max(1e-6, afr_actual))) # Equivalence ratio
    mean_piston = 2 * spec.stroke_m * rpm / 60                        # mean piston speed [m/s]
    if k_mean is None or k_mean < 10.0:
        c_t0 = 0.30                                                   # turbulence intensity fraction
        c_t = c_t0 *(1 + 0.15 * mean_piston / 10)                     # intensity increase for higher speeds
        u_prime = c_t * mean_piston                                   # u' 
        k_mean = 1.5 * u_prime**2                                     # k = 3/2 u'^2
    # Laminar/turbulent flame speed
    T_u = max(300.0, T_u_mean if T_u_mean else T_soc)
    p_u = max(8e4, p_soc)
    S_L = laminar_speed(fuel, T_u, p_u, phi)
    if t_length_mean is None: # Turbulence length scale
        t_length_mean = 0.4 * spec.bore_m
    S_T = turbulent_speed(S_L, k_mean, t_length_mean)
    L_char = 0.26 * spec.bore_m                     # Flame travel
    omega = rpm * 2*np.pi / 60.0                    # [rad/s]
    tau_burn = L_char / max(0.05, S_T)              # [s]
    theta_burn_rad = omega * tau_burn               # [rad]
    theta_10_90_rad = 0.9 * theta_burn_rad
    # Given a,m, compute Δθ (Wiebe duration)
    def xi(y):  # normalized crank for MFB=y
        return (-np.log(1.0 - y) / a)**(1.0/(m+1.0))
    xi10, xi90 = xi(0.10), xi(0.90)
    Delta_theta_rad = theta_10_90_rad / max(1e-6, (xi90 - xi10))
    soc_rad = np.deg2rad(-10.0)
    eoc_rad = soc_rad + Delta_theta_rad
    soc_rad = np.deg2rad(-10.0)
    eoc_rad = soc_rad + Delta_theta_rad
    # reindex burn window with the new SOC/EOC
    i_soc = int(np.argmin(np.abs(crank_angle - soc_rad)))
    i_eoc = int(np.argmin(np.abs(crank_angle - eoc_rad)))
    delta = eoc_rad - soc_rad
    if i_eoc >= i_evo:
        i_eoc = i_evo - 1
        eoc_rad = crank_angle[i_eoc]
        delta   = eoc_rad - soc_rad
    if i_eoc <= i_soc:
        i_eoc = min(i_soc + 3, i_evo - 1, len(crank_angle) - 2)
        eoc_rad = crank_angle[i_eoc]
        delta   = eoc_rad - soc_rad
    # CA10/50/90
    def ca_at_mfb(y):
        x = (-np.log(1.0 - y) / a)**(1.0 / (m + 1.0))
        return soc_rad + x * delta
    ca10_rad = ca_at_mfb(0.10)
    ca50_rad = ca_at_mfb(0.50)
    ca90_rad = ca_at_mfb(0.90)
    ca10_deg, ca50_deg, ca90_deg = map(np.degrees, (ca10_rad, ca50_rad, ca90_rad))
    # COMBUSTION
    P_combustion, T_combustion, mfb_list = [], [], []
    for i in range(i_soc, i_eoc):
        theta = crank_angle[i]
        x = (theta - soc_rad) / max(1e-12, delta)
        x = np.clip(x, 0.0, 1.0)
        mfb = 1.0 - np.exp(-a * x**(m+1))
        dmfb_dtheta = a * (m+1) * x**m * np.exp(-a * x**(m+1)) / max(1e-12, delta)
        dQ_chem = Q_tot * dmfb_dtheta * dtheta
        dU = dQ_chem * 0.9 - P_current * (V[i+1] - V[i])  # dU = dQ - dW
        cv_loc = cv_mix(T_current, mfb)
        dT_combustion = dU / (m_trapped * cv_loc)
        T_current += dT_combustion
        V_current = V[i+1]
        P_current = m_trapped * gas_constant * T_current / V_current
        T_combustion.append(T_current)
        P_combustion.append(P_current)
        mfb_list.append(mfb)
    # EXPANSION STROKE
    P_eoc = P_current   
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
    df.loc[i_soc:i_eoc - 1, 'Pressure (bar)'] = np.array(P_combustion) / 1e5
    df.loc[i_eoc:i_evo, 'Pressure (bar)'] = P_expansion / 1e5
    df.loc[i_evo:i_ivo, 'Pressure (bar)'] = P_blowdown / 1e5
    df.loc[i_ivc:i_soc, 'Temperature (K)']  = T_compression
    df.loc[i_soc:i_eoc - 1, 'Temperature (K)']  = np.array(T_combustion)
    df.loc[i_eoc:i_evo, 'Temperature (K)']  = T_expansion
    df.loc[i_evo:i_ivo, 'Temperature (K)']  = T_blowdown
    df.loc[i_soc:i_eoc - 1, 'Mass Fraction Burned'] = np.array(mfb_list)
    # p–V + IMEP
    P_pa = (df['Pressure (bar)'].to_numpy() * 1e5)
    V_m3 = df['Volume (m3)'].to_numpy()
    mask = ~np.isnan(P_pa)
    W_cyl = np.trapezoid(P_pa[mask], V_m3[mask])
    imep_gross = W_cyl / V_displacement
    fmep = (0.8 + 0.00050 * mean_piston + 0.00550 * (mean_piston) ** 2) * 1e5
    pmep = (0.03 + 0.000035 * rpm) * 1e5
    bmep = imep_gross - fmep - pmep
    Vd_total = V_displacement * spec.n_cylinder
    Torque_Nm = bmep * Vd_total / (4*np.pi)  # 4π for 4-stroke
    omega = rpm * 2*np.pi / 60.0
    Power_kW = Torque_Nm * omega / 1000.0
    bsfc = mFuelpersec * 3600 / max(Power_kW, 1e-9)
    # PLOTTING
    if plot:
        plt.figure()
        plt.plot(V_m3[mask], P_pa[mask]/1e5)
        plt.xlabel('Volume [m³]'); plt.ylabel('Pressure [bar]')
        plt.title('p–V Loop (single cylinder)'); plt.grid(True); plt.show()

        plt.figure(figsize=(10,6))
        plt.subplot(3,1,1)
        plt.plot(df['Crank Angle (deg)'], df['Pressure (bar)'])
        for ca_deg, label in [(np.degrees(ca10_rad),'CA10'),
                              (np.degrees(ca50_rad),'CA50'),
                              (np.degrees(ca90_rad),'CA90')]:
            plt.axvline(ca_deg, ls='--', lw=1, c='k', alpha=0.7)
            ymax = plt.gca().get_ylim()[1]; plt.text(ca_deg, 0.95*ymax, label,
                                                     rotation=90, va='top', ha='right', fontsize=9)
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
        plt.plot([np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)],
                 [0.10, 0.50, 0.90], 'ko', ms=4)
        plt.xlabel('Crank Angle [deg]'); plt.ylabel('MFB [-]'); plt.grid(True)
        plt.tight_layout(); plt.show()
        return

    # RESULTS
    if return_dic:
        out = {
            "imep_gross_pa": imep_gross,
            "bmep_pa": bmep,
            "fmep_pa": fmep,
            "pmep_pa": pmep,
            "torque_nm": Torque_Nm,
            "power_kw": Power_kW,
            "m_air_per_cycle": mAirpercycle,    # per cyl, per cycle
            "m_fuel_per_cycle": mFuelpercycle,  # per cyl, per cycle
            "ca10_deg": ca10_deg,
            "ca50_deg": ca50_deg,
            "ca90_deg": ca90_deg,
            "pmax_bar": np.nanmax(df['Pressure (bar)'].to_numpy()),
            "tmax_k":   np.nanmax(df['Temperature (K)'].to_numpy()),
            "S_L_mps": float(S_L),
            "S_T_mps": float(S_T),
            "Delta_theta_deg": float(np.degrees(Delta_theta_rad)),
            "soc_deg": float(np.degrees(soc_rad)),
            "eoc_deg": float(np.degrees(eoc_rad)),
            "phi": float(phi),
            "k_mean": float(k_mean),
            "ell_t_m": float(t_length_mean),
            "T_u_K": float(T_u),
            "p_u_bar": float(p_u / 1e5),
        }
        return out

#%% EMISSIONS (kept as-is)
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
    """
    lam = AFR / 14.7
    lam = max(0.5, min(1.6, lam))

    # CO (g/kg fuel)
    if lam >= 1.0:
        EI_CO = 3.0 * (1 + 4.0*(lam - 1.0))
    else:
        EI_CO = 5.0 + 800.0*(1.0 - lam)**2
    EI_CO = min(EI_CO, 400.0)

    # HC (g/kg fuel)
    rich_term = 200.0*(max(0.0, 1.0 - lam))**2
    lean_term = 120.0*(max(0.0, lam - 1.15))**2
    incomp_term = 50.0*(1.0 - max(0.0, min(1.0, comb_eff)))
    EI_HC = 2.0 + rich_term + lean_term + incomp_term
    EI_HC = min(EI_HC, 300.0)

    # NOx (g/kg fuel)
    lam_peak = 1.05
    sigma = 0.09
    peak_noX = 18.0 * (load_frac**0.7)
    EI_NOx = peak_noX * np.exp(-0.5*((lam - lam_peak)/sigma)**2)

    # CO2 (g/kg fuel)
    EI_CO2 = ei_co2_g_per_kg

    gps_CO2 = EI_CO2 * mDotFuel
    gps_CO  = EI_CO  * mDotFuel
    gps_NOx = EI_NOx * mDotFuel
    gps_HC  = EI_HC  * mDotFuel

    return {'CO2': gps_CO2, 'CO': gps_CO, 'NOx': gps_NOx, 'HC': gps_HC}