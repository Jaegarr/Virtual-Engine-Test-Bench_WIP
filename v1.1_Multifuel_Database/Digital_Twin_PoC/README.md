# 🧪 Digital Twin Proof-of-Concept Demonstrations

---

## 🎯 Overview
This repository contains three proof-of-concept demonstrations showcasing the author’s prior independent work toward an integrated **multi-fidelity digital twin** architecture combining:
- physics-based combustion modelling  
- machine-learning residual correction  
- CFD-derived turbulence and temperature coupling  

All simulations were conducted using the author’s **Virtual Engine Test Bench (VTEB)** Python framework.

---

## 🔹 Demo 1 — Multi-Fuel Combustion Behaviour

**Objective:**  
Evaluate the combustion characteristics of **gasoline, hydrogen, ammonia**, and a **10 % H₂–NH₃ blend** at 3000 rpm under wide-open-throttle (WOT) conditions to assess lean-limit behaviour and fuel-specific IMEP trends.

**Methodology:**
- Single-zone Wiebe-function combustion model (Python).  
- Stoichiometric-to-lean operation (λ = 0.8 – 1.5).  
- Fixed burn duration beyond λ ≥ 1.3 for numerical stability.  
- Fuel-specific thermochemical and laminar flame-speed correlations.  

**Results Summary:**
- Hydrogen achieved the highest IMEP (~15.6 bar at λ = 0.8) and stable operation up to λ = 1.5.  
- Gasoline produced intermediate IMEP (~13.4 bar) with a near-linear lean drop-off.  
- Ammonia yielded the lowest IMEP (~13.2 bar) due to slow kinetics and high ignition-energy demand.  
- A 10 % H₂–NH₃ blend increased IMEP by ~30 % vs pure NH₃, demonstrating hydrogen-stabilisation effects.  
- IMEP hierarchy correlated directly with flame-speed ranking (H₂ > H₂–NH₃ > Gasoline > NH₃).

**Outputs:**  
`Demo_1_Multifuel_Combustion.png` — IMEP vs λ (dashed region = fixed burn duration).

---

## 🔹 Demo 2 — Machine-Learning Grey-Box Correction

**Objective:**  
Demonstrate how machine learning can refine a physics-based combustion model by applying small, interpretable residual corrections derived from dyno data.

**Methodology:**
- Baseline single-zone model trained against steady WOT torque sweep (1000–7000 rpm).  
- Multilayer Perceptron (MLP) learns residuals for:  
  - Lambda (λ) offset,  
  - FMEP offset (mechanical-loss correction),  
  - Start-of-Combustion (SOC) timing.  
- SPSA-based optimiser enforces smoothness and monotonicity constraints.  

**Results Summary:**
- ML correction reduced torque deviation vs dyno from >5 % to <1 %.  
- Improved correlation of predicted vs measured IMEP across the RPM range.  
- Demonstrated feasibility of combining physical interpretability with data-driven residual learning.

**Outputs:**  
`Demo_2_ML_Correction.png` — Torque comparison: Baseline vs ML-Corrected vs Dyno.

---

## 🚀 Demo 3 — CFD-Informed Combustion Adjustment

**Objective:**  
Demonstrate how coupling CFD-derived mean unburned-gas temperature (**T̄₍ᵤ₎**) and turbulence kinetic energy (**k̄**) fields influences combustion behaviour within the Virtual Engine Test Bench.

**Methodology:**
- Steady-state *OpenFOAM* case at **3000 rpm / WOT** used to extract volume-averaged **T̄₍ᵤ₎** and **k̄**.  
- Interpolated map of **T̄₍ᵤ₎** and **k̄** applied across 1000–7000 rpm.  
- `combustion_Wiebe()` modified to accept CFD-derived inputs:  
  - **T̄₍ᵤ₎** affects initial charge temperature,  
  - **k̄** scales turbulent flame speed and burn duration via √k̄ correlation.  
- Simulation re-run to visualise qualitative impact on combustion phasing and flame-speed behaviour.  

**Results Summary:**
- CFD coupling produced smoother and faster combustion at mid-range speeds due to elevated turbulence intensity.  
- A visible increase in **turbulent flame speed (S_T)** occurred until reaching the clipping limit (**S_T ≈ 60 m/s**).  
- The exercise is **not a validation study**; it purely demonstrates the integration workflow and physical influence of turb
