import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Calibration as cal
from Engine_Database import EngineSpec
from Fuel_Database import FuelSpec, Fuels

# Laminar & turbulent flame speed models
def laminar_speed(
    fuel: FuelSpec,
    T_unburned_K: float,          # unburned-gas temperature [K]
    p_unburned_Pa: float,         # unburned-gas pressure [Pa]
    phi: float,                    # equivalence ratio [-]
    S_ref_m_per_s: float = None,  # optional override of S_L at 300 K, 1 atm
    temp_exp_alpha: float = 2.0,       # S_L ∝ T^alpha   (generic)
    press_exp_beta: float = -0.25,     # S_L ∝ p^beta    (generic)
    phi_peak: float = 1.1,             # peak of S_L(ϕ) (gasolineish)
    phi_width: float = 0.35            # width of the ϕ parabola
) -> float:
    """
    Minimal laminar flame-speed correlation:
        S_L ≈ S_L,300K,1atm * (T/300)^a * (p/1 atm)^β * ϕ-shape
    Keeps S_L within sensible floors to avoid numerical issues.
    """
    S0 = fuel.S_L_300K_m_per_s if S_ref_m_per_s is None else S_ref_m_per_s
    T_term = (T_unburned_K / 300.0)**temp_exp_alpha
    p_term = (p_unburned_Pa / 101_325.0)**press_exp_beta
    # simple parabola in phi around phi_peak
    phi_shape = max(0.15, 1.0 - ((phi - phi_peak) / phi_width) ** 2)
    return max(0.02, S0 * T_term * p_term * phi_shape)  # floor ~2 cm/s
def turbulent_speed(
    S_L_m_per_s: float,
    k_turb_m2_per_s2: float,  # turbulent kinetic energy k [m^2/s^2]
    ell_turb_m: float,        # turbulence integral length [m]
    nu_kinematic_m2_per_s: float = 1.7e-5,  # (unused placeholder)
    C_t: float = 2.0,
    n_exp: float = 1.0,
    m_exp: float = 0.3
) -> float:
    """
    Bradley-style wrinkled-flame model:
        u' = sqrt(2/3 * k)
        δ_L ~ a_th / S_L
        S_T = S_L * (1 + C_t * (u'/S_L)^n * (ell_t/δ_L)^m)

    Notes:
    - We clamp δ_L to avoid vanishingly small thickness.
    - We clamp S_T to a sensible range for stability.
    """
    u_prime_m_per_s = np.sqrt(max(0.0, 2.0/3.0 * k_turb_m2_per_s2))
    alpha_thermal_m2_per_s = 2.0e-5  # rough thermal diffusivity of unburned mix
    # Clamp laminar thickness to ≥ 0.2 mm
    delta_L_m = max(2.0e-4, alpha_thermal_m2_per_s / max(0.02, S_L_m_per_s))
    scale = (u_prime_m_per_s / max(0.02, S_L_m_per_s))**n_exp
    if m_exp != 0.0:
        scale *= (ell_turb_m / delta_L_m)**m_exp
    S_T_m_per_s = S_L_m_per_s * (1.0 + C_t * scale)
    return float(np.clip(S_T_m_per_s, max(S_L_m_per_s*1.05, 5.0), 60.0))
# Knock proxy (Livengood–Wu integral with a very simple ignition-delay law)
# tau_ignition_s ≈ A * p_bar^-n * exp(E/R/T_endgas)
A_tau = 1.2e-3
n_tau = 1.0
Ea_over_R = 3800.0  # Kelvin (activation energy over R)
def tau_ignition_s(p_bar: float, T_endgas_K: float) -> float:
    T_endgas_K = max(300.0, T_endgas_K)
    return A_tau * (max(p_bar, 1.0) ** (-n_tau)) * np.exp(Ea_over_R / T_endgas_K)
# MAIN: Single-zone Wiebe-based combustion model with light heat transfer
def combustion_Wiebe(
    spec: EngineSpec,
    rpm: float,
    throttle: float,
    ve: float,
    fuel: FuelSpec,
    k_turb_m2_per_s2: float = 5.0,    # turbulence intensity (from CFD or guess)
    T_u_mean_K: float = 330.0,        # mean unburned-gas temp at SOC (if known)
    ell_turb_m: float | None = None,  # turbulence integral length [m]
    R_gas_J_per_kgK: float = 287.0,   # gas constant [J/(kg·K)]
    T_ivc_K: float = 330.0,           # intake-valve-closure temp [K]
    wiebe_a: float = 5.0, wiebe_m: float = 2.0,
    combustion_eff: float = 0.98,     # fraction of LHV released
    n_poly_comp: float = 1.34, n_poly_exp: float = 1.26,
    soc_offset_deg: float = 0.0,      
    fmep_offset_Pa: float = 0.0,
    target_lambda: float | None = None,      
    plot: bool = True, return_dic: bool = False
):
    """
    Single-cylinder cycle with:
    - polytropic compression/expansion
    - Wiebe heat-release window sized by a simple S_T-based timescale
    - Woschni-lite convective heat transfer (keeps Tmax realistic)
    - Livengood - Wu knock index (proxy)
    """
    # 1) Crank-angle grid [rad]
    theta_rad = np.linspace(-np.pi, np.pi, 1441)
    dtheta_rad = theta_rad[1] - theta_rad[0]
    # 2) Valve events & burn window
    IVO_rad = np.deg2rad(130.0)
    IVC_rad = np.deg2rad(-110.0)
    EVO_rad = np.deg2rad(110.0)
    EVC_rad = np.deg2rad(-160.0)
    SoC_rad = np.deg2rad(-8.0 + soc_offset_deg)  # start of combustion, EOC is computed later from the S_T-based duration 

    idx_IVO = int(np.argmin(np.abs(theta_rad - IVO_rad)))
    idx_IVC = int(np.argmin(np.abs(theta_rad - IVC_rad)))
    idx_EVO = int(np.argmin(np.abs(theta_rad - EVO_rad)))
    idx_EVC = int(np.argmin(np.abs(theta_rad - EVC_rad)))
    # 3) Temperature-dependent cv [J/(kg·K)]
    T_knots = np.array([300, 600, 1000, 1500, 2000, 2500, 3000, 3500])
    cv_u_knots = np.array([718, 740, 820, 900, 960, 1000, 1030, 1050])   # unburned
    cv_b_knots = np.array([740, 780, 880, 1000, 1100, 1150, 1200, 1250]) # burned
    def cv_unburned(T_K: float) -> float:
        T_K = np.clip(T_K, 300.0, 3500.0)
        return np.interp(T_K, T_knots, cv_u_knots)
    def cv_burned(T_K: float) -> float:
        T_K = np.clip(T_K, 300.0, 3500.0)
        return np.interp(T_K, T_knots, cv_b_knots)
    def cv_mix(T_K: float, mfb: float) -> float:
        # linear blend by mass fraction burned (0..1)
        return (1.0 - mfb) * cv_unburned(T_K) + mfb * cv_burned(T_K)
    # 4) Geometry
    V_disp_m3 = np.pi * (spec.bore_m**2 / 4.0) * spec.stroke_m
    V_clear_m3 = V_disp_m3 / (spec.compression_ratio - 1.0)
    crank_radius_m = spec.stroke_m / 2.0
    cyl_area_m2 = np.pi * spec.bore_m**2 / 4.0
    # 5) Heat transfer helpers (Woschni-lite)
    HT_SCALE = 0.05
    T_wall_K = 520.0  
    def gas_velocity_mean(Up_m_per_s, p_Pa, p_mot_Pa):
        C1 = 3.5         
        return C1 * Up_m_per_s
    def h_woschni_W_per_m2K(p_Pa, T_K, bore_m, w_m_per_s):
        h_nom = 3.26 * (bore_m**-0.2) * (max(p_Pa,1e5)**0.8) * (max(T_K,300)**-0.55) * (max(w_m_per_s,0.1)**0.8)
        return HT_SCALE * h_nom
    def area_from_volume_m2(V_m3: float, bore_m: float) -> float:
        # Rough A ~ k * V^(2/3); choose k so near-TDC area ≈ half piston crown
        A_piston_half = np.pi * (bore_m**2) / 2.0
        k = A_piston_half / ((V_clear_m3 + 1e-12)**(2.0/3.0))
        return k * (V_m3**(2.0/3.0))
    # 6) Kinematics: volume vs crank angle
    piston_pos_m = (crank_radius_m * (1 - np.cos(theta_rad)) + (crank_radius_m**2) / (2.0 * spec.conrod_m) * (1 - np.cos(2.0 * theta_rad)))
    V_m3 = V_clear_m3 + cyl_area_m2 * piston_pos_m
    dV_dtheta_m3_per_rad = np.gradient(V_m3, theta_rad)
    # 7) Masses at IVC
    p_ivc_Pa = 20e3 + throttle * (100e3 - 20e3)    
    rho_ivc_kg_per_m3 = p_ivc_Pa / (R_gas_J_per_kgK * T_ivc_K)
    m_air_per_cycle_kg = rho_ivc_kg_per_m3 * V_m3[idx_IVC] * ve
    if target_lambda is not None:
        lam = float(target_lambda)
        afr_target = fuel.AFR_stoich * lam
    else:
        afr_target = cal.get_target_AFR(rpm, fuel=fuel)     
    m_fuel_per_cycle_kg = m_air_per_cycle_kg / afr_target
    m_trapped_kg = m_air_per_cycle_kg + m_fuel_per_cycle_kg
    # Engine-level flows
    cyl_per_sec = spec.n_cylinder * rpm / 120.0
    m_air_per_s_kg = m_air_per_cycle_kg * cyl_per_sec
    m_fuel_per_s_kg = m_fuel_per_cycle_kg * cyl_per_sec
    # 8) Compression to SoC (polytropic)
    idx_SOC = int(np.argmin(np.abs(theta_rad - SoC_rad)))
    V_comp = V_m3[idx_IVC:idx_SOC+1]
    P_comp_Pa = p_ivc_Pa * (V_m3[idx_IVC] / V_comp) ** n_poly_comp
    T_comp_K = (P_comp_Pa * V_comp) / (m_trapped_kg * R_gas_J_per_kgK)

    P_curr_Pa = P_comp_Pa[-1]
    T_curr_K = T_comp_K[-1]
    p_SoC_Pa = P_comp_Pa[-1]
    T_SoC_K = T_comp_K[-1]

    phi = max(0.0, min(2.0, fuel.AFR_stoich / max(1e-6, afr_target)))  # ϕ = AFR_stoich / AFR_actual
    # turbulence
    mean_piston_speed_m_per_s = 2.0 * spec.stroke_m * rpm / 60.0
    if k_turb_m2_per_s2 is None or k_turb_m2_per_s2 < 10.0:
        # Base coefficient
        c_t0_base = 0.30
        # Add 20% boost at very low RPM, taper to baseline by 3000 rpm
        if rpm < 2000:
            c_t_mult = 1.20
        elif rpm < 3000:
            c_t_mult = 1.20 - 0.20 * (rpm - 2000) / 1000.0
        else:
            c_t_mult = 1.0
        c_t0 = c_t0_base * c_t_mult
        c_t = c_t0 * (1 + 0.15 * mean_piston_speed_m_per_s / 10.0) # Effective turbulence scaling with mean piston speed
        u_prime = c_t * mean_piston_speed_m_per_s
        k_turb_m2_per_s2 = 1.5 * u_prime**2  # k = 3/2 u'^2
    # 9) Flame speeds & Wiebe duration
    T_unburned_K = max(300.0, T_u_mean_K if T_u_mean_K else T_SoC_K)
    p_unburned_Pa = max(8e4, p_SoC_Pa)
    S_L = laminar_speed(fuel, T_unburned_K, p_unburned_Pa, phi)
    if ell_turb_m is None:
        ell_turb_m = 0.4 * spec.bore_m  
    S_T = turbulent_speed(S_L, k_turb_m2_per_s2, ell_turb_m)

    L_char_m = 0.26 * spec.bore_m                    # flame travel scale
    omega_rad_per_s = rpm * 2.0 * np.pi / 60.0
    tau_burn_s = L_char_m / max(0.05, S_T)           # keep a floor on S_T
    theta_burn_rad = omega_rad_per_s * tau_burn_s
    theta_10_90_rad = 0.9 * theta_burn_rad
    # map 10–90% MFB duration to Wiebe Δθ for (a,m)
    def _xi(mfb: float) -> float:
        return (-np.log(1.0 - mfb) / wiebe_a) ** (1.0 / (wiebe_m + 1.0))
    xi10, xi90 = _xi(0.10), _xi(0.90)
    denom = max(1e-9, (xi90 - xi10))                  # FIX: guard divide-by-zero
    Delta_theta_rad = theta_10_90_rad / denom
    # Burn window [SOC..EOC] in crank angle
    EoC_rad = SoC_rad + Delta_theta_rad
    idx_SOC = int(np.argmin(np.abs(theta_rad - SoC_rad)))
    idx_EOC = int(np.argmin(np.abs(theta_rad - EoC_rad)))
    burn_span_rad = EoC_rad - SoC_rad
    # ensure EOC is within EVO and after SOC
    if idx_EOC >= idx_EVO:
        idx_EOC = idx_EVO - 1
        EoC_rad = theta_rad[idx_EOC]
        burn_span_rad = EoC_rad - SoC_rad
    if idx_EOC <= idx_SOC:
        idx_EOC = min(idx_SOC + 3, idx_EVO - 1, len(theta_rad) - 2)
        EoC_rad = theta_rad[idx_EOC]
        burn_span_rad = EoC_rad - SoC_rad
    # 10/50/90 crank angles
    def crank_at_mfb(mfb: float) -> float:
        x = (-np.log(1.0 - mfb) / wiebe_a) ** (1.0 / (wiebe_m + 1.0))
        return SoC_rad + x * burn_span_rad

    ca10_rad = crank_at_mfb(0.10)
    ca50_rad = crank_at_mfb(0.50)
    ca90_rad = crank_at_mfb(0.90)
    ca10_deg, ca50_deg, ca90_deg = map(np.degrees, (ca10_rad, ca50_rad, ca90_rad))
    # 10) Total heat release [J]
    Q_total_J = m_fuel_per_cycle_kg * combustion_eff * fuel.LHV_MJ_per_kg * 1e6
    qchem_int_J = 0.0   # ∫ dQ_chem
    qht_int_J   = 0.0   # ∫ dQ_ht
    # 11) Combustion loop (with heat transfer & knock integral)
    P_comb_Pa, T_comb_K, mfb_list = [], [], []
    knock_index = 0.0
    T_endgas_K = T_comp_K[-1]             
    p_endgas_bar = P_comp_Pa[-1] / 1e5
    for i in range(idx_SOC, idx_EOC):
        theta_i = theta_rad[i]
        x_norm = np.clip((theta_i - SoC_rad) / max(1e-12, burn_span_rad), 0.0, 1.0)

        mfb = 1.0 - np.exp(-wiebe_a * x_norm ** (wiebe_m + 1.0))
        dmfb_dtheta = (wiebe_a * (wiebe_m + 1.0) * x_norm ** wiebe_m
                       * np.exp(-wiebe_a * x_norm ** (wiebe_m + 1.0))
                       / max(1e-12, burn_span_rad))

        dQ_chem_J = Q_total_J * dmfb_dtheta * dtheta_rad
        qchem_int_J += dQ_chem_J
        dt_s = float(dtheta_rad / omega_rad_per_s)

        # Heat transfer (Woschni-lite)
        j = i - idx_IVC
        p_mot_Pa = P_comp_Pa[j] if 0 <= j < len(P_comp_Pa) else P_curr_Pa
        w_gas_m_per_s = gas_velocity_mean(mean_piston_speed_m_per_s, P_curr_Pa, p_mot_Pa)
        h_W_per_m2K = h_woschni_W_per_m2K(P_curr_Pa, T_curr_K, spec.bore_m, w_gas_m_per_s)
        A_ht_m2 = area_from_volume_m2(V_m3[i], spec.bore_m)
        Qdot_ht_W = h_W_per_m2K * A_ht_m2 * max(T_curr_K - T_wall_K, 0.0)
        dQ_ht_J = Qdot_ht_W * dt_s
        qht_int_J   += dQ_ht_J
        # First law: dU = dQ_chem - dQ_ht - p dV
        dU_J = dQ_chem_J - dQ_ht_J - P_curr_Pa * (V_m3[i+1] - V_m3[i])
        cv_now = cv_mix(T_curr_K, mfb)  # J/(kg·K)
        dT_K = dU_J / (m_trapped_kg * cv_now)
        T_curr_K += dT_K
        P_curr_Pa = m_trapped_kg * R_gas_J_per_kgK * T_curr_K / V_m3[i+1]

        T_comb_K.append(T_curr_K)
        P_comb_Pa.append(P_curr_Pa)
        mfb_list.append(mfb)
        # Knock proxy
        gamma_u = 1.35
        T_endgas_K *= (P_curr_Pa/1e5 / max(p_endgas_bar, 1e-3)) ** ((gamma_u - 1.0) / gamma_u)
        p_endgas_bar = P_curr_Pa / 1e5
        tau_s = tau_ignition_s(p_endgas_bar, T_endgas_K)
        knock_index += dt_s / max(tau_s, 1e-9)
    # 12) Expansion (polytropic)
    P_eoc_Pa = P_curr_Pa
    V_eoc_m3 = V_m3[idx_EOC]
    V_exp = V_m3[idx_EOC:idx_EVO+1]
    P_exp_Pa = P_eoc_Pa * (V_eoc_m3 / V_exp) ** n_poly_exp
    T_exp_K = P_exp_Pa * V_exp / (m_trapped_kg * R_gas_J_per_kgK)
    # 13) Blowdown (very compact unsteady mass release surrogate)
    theta_bd = theta_rad[idx_EVO:idx_IVO+1]
    V_bd = V_m3[idx_EVO:idx_IVO+1]

    P_exhaust_Pa = 1.05e5
    T_exhaust_target_K = 1150.0
    V_ivo_m3 = V_m3[idx_IVO]
    m_target_kg = P_exhaust_Pa * V_ivo_m3 / (R_gas_J_per_kgK * T_exhaust_target_K)

    P_evo_Pa = P_exp_Pa[-1]
    T_evo_K = T_exp_K[-1]
    m_evo_kg = m_trapped_kg

    omega = omega_rad_per_s
    delta_m_kg = m_evo_kg - m_target_kg
    den = max(IVO_rad - EVO_rad, 1e-12)
    alpha = np.clip((theta_bd - EVO_rad) / den, 0.0, 1.0)
    beta_shape = 0.30
    S_rel = (1.0 - np.exp(-alpha / beta_shape)) / (1.0 - np.exp(-1.0 / beta_shape))
    m_bd_kg = m_evo_kg - delta_m_kg * S_rel
    m_bd_kg = np.minimum.accumulate(m_bd_kg)

    P_bd_Pa = np.empty_like(theta_bd)
    T_bd_K = np.empty_like(theta_bd)
    T_bd_K[0] = T_evo_K
    P_bd_Pa[0] = m_bd_kg[0] * R_gas_J_per_kgK * T_bd_K[0] / V_bd[0]

    for k in range(len(theta_bd) - 1):
        dtheta_k = theta_bd[k+1] - theta_bd[k]
        dt_k = dtheta_k / omega

        dVdtheta_k = (V_bd[k+1] - V_bd[k]) / dtheta_k
        dVdt_k = dVdtheta_k * omega

        dm_dtheta_k = (m_bd_kg[k+1] - m_bd_kg[k]) / dtheta_k
        m_dot_out_k = -dm_dtheta_k * omega

        P_k = m_bd_kg[k] * R_gas_J_per_kgK * T_bd_K[k] / V_bd[k]
        cv_bd = cv_burned(T_bd_K[k])
        dTdt_k = (-P_k * dVdt_k - R_gas_J_per_kgK * T_bd_K[k] * m_dot_out_k) / (m_bd_kg[k] * cv_bd)

        T_bd_K[k+1] = max(300.0, T_bd_K[k] + dTdt_k * dt_k)
        P_bd_Pa[k+1] = m_bd_kg[k+1] * R_gas_J_per_kgK * T_bd_K[k+1] / V_bd[k+1]
    # 14) DataFrame
    df = pd.DataFrame({
        "Crank Angle (deg)": np.degrees(theta_rad),
        "Volume (m3)": V_m3,
        "dVdtheta (m3/rad)": dV_dtheta_m3_per_rad,
        "Pressure (bar)": np.nan,
        "Temperature (K)": np.nan,
        "Mass Fraction Burned": np.nan
    })
    df.loc[idx_IVC:idx_SOC, "Pressure (bar)"] = P_comp_Pa / 1e5
    df.loc[idx_SOC:idx_EOC-1, "Pressure (bar)"] = np.array(P_comb_Pa) / 1e5
    df.loc[idx_EOC:idx_EVO, "Pressure (bar)"] = P_exp_Pa / 1e5
    df.loc[idx_EVO:idx_IVO, "Pressure (bar)"] = P_bd_Pa / 1e5
    df.loc[idx_IVC:idx_SOC, "Temperature (K)"] = T_comp_K
    df.loc[idx_SOC:idx_EOC-1, "Temperature (K)"] = np.array(T_comb_K)
    df.loc[idx_EOC:idx_EVO, "Temperature (K)"] = T_exp_K
    df.loc[idx_EVO:idx_IVO, "Temperature (K)"] = T_bd_K
    df.loc[idx_SOC:idx_EOC-1, "Mass Fraction Burned"] = np.array(mfb_list)
    # 15) Work/IMEP/BMEP/Torque/Power
    P_Pa = (df["Pressure (bar)"].to_numpy() * 1e5)
    V_curve = df["Volume (m3)"].to_numpy()
    mask = ~np.isnan(P_Pa)
    W_cyc_J = np.trapezoid(P_Pa[mask], V_curve[mask])  # area under p–V
    imep_gross_Pa = W_cyc_J / V_disp_m3
    burn_10_90_deg = float(np.degrees(ca90_rad - ca10_rad))
    w_ind_J_per_cycle = float(W_cyc_J)
    eta_ind_pct = 100.0 * w_ind_J_per_cycle / max(Q_total_J, 1e-12)   # indicated efficiency [%]

    # simple friction & pumping maps (SI units)
    fmep_base_Pa = (0.1 + 0.00055 * mean_piston_speed_m_per_s + 0.005 * (mean_piston_speed_m_per_s)**2) * 1e5 
    fmep_Pa = max(fmep_base_Pa + fmep_offset_Pa, 0.05e5)
    pmep_Pa = (0.03 + 0.000007 * rpm) * 1e5
    bmep_Pa = imep_gross_Pa - fmep_Pa - pmep_Pa

    Vd_total_m3 = V_disp_m3 * spec.n_cylinder
    torque_Nm = bmep_Pa * Vd_total_m3 / (4.0 * np.pi)  # 4π for 4-stroke
    power_kW = torque_Nm * (omega_rad_per_s) / 1000.0
    bsfc_g_per_kWh = (m_fuel_per_s_kg * 3600.0 * 1000.0) / max(power_kW, 1e-12)
    # 16) Plots
    if plot:
        plt.figure()
        plt.plot(V_curve[mask], P_Pa[mask]/1e5)
        plt.xlabel("Volume [m³]"); plt.ylabel("Pressure [bar]")
        plt.title("p–V Loop (single cylinder)"); plt.grid(True); plt.show()
        plt.figure(figsize=(10,6))
        # Pressure trace
        plt.subplot(3,1,1)
        plt.plot(df["Crank Angle (deg)"], df["Pressure (bar)"])
        for ca_deg, label in [(np.degrees(ca10_rad), "CA10"),
                              (np.degrees(ca50_rad), "CA50"),
                              (np.degrees(ca90_rad), "CA90")]:
            plt.axvline(ca_deg, ls="--", lw=1, c="k", alpha=0.7)
            ymax = plt.gca().get_ylim()[1]
            plt.text(ca_deg, 0.95*ymax, label, rotation=90, va="top", ha="right", fontsize=9)
        plt.ylabel("Pressure [bar]"); plt.grid(True)
        # Temperature trace
        plt.subplot(3,1,2)
        plt.plot(df["Crank Angle (deg)"], df["Temperature (K)"])
        for ca_deg in (np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)):
            plt.axvline(ca_deg, ls="--", lw=1, c="k", alpha=0.5)
        plt.ylabel("Temperature [K]"); plt.grid(True)
        # Burn fraction
        plt.subplot(3,1,3)
        plt.plot(df["Crank Angle (deg)"], df["Mass Fraction Burned"])
        for ca_deg in (np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)):
            plt.axvline(ca_deg, ls="--", lw=1, c="k", alpha=0.5)
        for y in (0.10, 0.50, 0.90): plt.axhline(y, ls=":", lw=1, c="gray", alpha=0.6)
        plt.plot([np.degrees(ca10_rad), np.degrees(ca50_rad), np.degrees(ca90_rad)],
                 [0.10, 0.50, 0.90], "ko", ms=4)
        plt.xlabel("Crank Angle [deg]"); plt.ylabel("MFB [-]"); plt.grid(True)
        plt.tight_layout(); plt.show()
        return
    if return_dic:
        return {
        # performance / work
        "imep_gross_pa":        float(imep_gross_Pa),
        "bmep_pa":              float(bmep_Pa),
        "fmep_pa":              float(fmep_Pa),
        "pmep_pa":              float(pmep_Pa),
        "torque_nm":            float(torque_Nm),
        "power_kw":             float(power_kW),
        "bsfc_g_per_kwh":       float(bsfc_g_per_kWh),
        # mass (per cycle, per cylinder)
        "m_air_per_cycle_kg":   float(m_air_per_cycle_kg),
        "m_fuel_per_cycle_kg":  float(m_fuel_per_cycle_kg),
        # mixture / timings
        "lambda":               float(1 / phi),
        "phi":                  float(phi),
        "ca10_deg":             float(ca10_deg),
        "ca50_deg":             float(ca50_deg),
        "ca90_deg":             float(ca90_deg),
        "soc_deg":              float(np.degrees(SoC_rad)),
        "eoc_deg":              float(np.degrees(EoC_rad)),
        "burn_10_90_deg":       float(burn_10_90_deg),
        # in-cylinder peaks
        "pmax_bar":             float(np.nanmax(df["Pressure (bar)"].to_numpy())),
        "tmax_k":               float(np.nanmax(df["Temperature (K)"].to_numpy())),
        # flame speeds
        "S_L_m_per_s":          float(S_L),
        "S_T_m_per_s":          float(S_T),
        # turbulence / end-gas state used for correlations
        "k_turb_m2_per_s2":     float(k_turb_m2_per_s2),
        "ell_turb_m":           float(ell_turb_m),
        "T_unburned_K":         float(T_unburned_K),
        "p_unburned_bar":       float(p_unburned_Pa / 1e5),
        # knock proxy
        "knock_index":          float(knock_index),
        # energy(per cycle, per cylinder)
        "q_chem_kj_per_cycle":  float(qchem_int_J / 1000.0),
        "q_ht_kj_per_cycle":    float(qht_int_J   / 1000.0),
        "w_ind_kj_per_cycle":   float(w_ind_J_per_cycle / 1000.0),
        "eta_ind_percent":      float(eta_ind_pct),
    }
#%% EMISSIONS
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

